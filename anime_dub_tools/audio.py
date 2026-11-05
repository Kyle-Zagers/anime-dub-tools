from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from anime_dub_tools.models import DubError

DIALOGUE_RATE = 22050
BED_RATE = 44100
LINE_FADE_SECONDS = 0.010


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise DubError("ffmpeg and ffprobe must be on PATH.")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        tail = detail[-2000:]
        raise DubError(f"Command failed ({result.returncode}): {args[0]}\n{tail}")
    return result


def media_duration(path: Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise DubError(f"Could not read a duration from {path}") from exc


def stream_types(path: Path) -> set[str]:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def has_video_stream(path: Path) -> bool:
    return "video" in stream_types(path)


def extract_audio(episode: Path, destination: Path) -> None:
    """Resample to 44.1 kHz and keep every channel.

    A 5.1 source stays 5.1. The stereo pair is replaced later; the other
    channels are not folded into it here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(episode),
            "-vn",
            "-ar",
            str(BED_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def probe_audio(path: Path) -> tuple[int, int]:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise DubError(f"Could not read the channel layout of {path}")
    try:
        return int(lines[0]), int(lines[1])
    except ValueError as exc:
        raise DubError(f"Could not read the channel layout of {path}") from exc


def load_mono(path: Path, sample_rate: int = DIALOGUE_RATE) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise DubError(f"Could not read audio from {path}\n{detail}")
    raw = result.stdout
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return pcm / 32768.0


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = DIALOGUE_RATE) -> None:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 1:
        mono = mono[:, None]
    write_audio(path, mono, sample_rate)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float audio as (samples, channels) at the file's own rate."""
    pcm, sample_rate = _load_pcm(path)
    return pcm.astype(np.float32) / 32768.0, sample_rate


def _load_pcm(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, channels = probe_audio(path)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise DubError(f"Could not read audio from {path}\n{detail}")
    raw = result.stdout
    width = channels * 2
    if width and len(raw) % width:
        raw = raw[: len(raw) - (len(raw) % width)]
    if not raw:
        return np.zeros((0, channels), dtype="<i2"), sample_rate
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).copy()
    return pcm, sample_rate


def write_audio(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    import wave

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def pad_trim_2d(samples: np.ndarray, count: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if count <= 0:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)
    if audio.shape[0] == count:
        return audio
    if audio.shape[0] > count:
        return audio[:count].copy()
    out = np.zeros((count, audio.shape[1]), dtype=np.float32)
    out[: audio.shape[0]] = audio
    return out


def stereo_bed(mix: np.ndarray, vocals: np.ndarray) -> np.ndarray:
    """Original mix minus the vocal estimate on the stereo pair only.

    Channels after the first two are copied unchanged. This is not a
    dialogue/music/effects stem and it is not a mono downmix.
    """
    bed = np.asarray(mix, dtype=np.float32)
    if bed.ndim != 2 or bed.shape[1] < 1:
        raise DubError("The background mix must be a (samples, channels) array.")
    estimate = np.asarray(vocals, dtype=np.float32)
    if estimate.ndim == 1:
        estimate = estimate[:, None]
    pair_count = min(2, bed.shape[1])
    if estimate.shape[1] == 1 and pair_count == 2:
        estimate = np.repeat(estimate, 2, axis=1)
    estimate = pad_trim_2d(estimate[:, :pair_count], bed.shape[0])
    output = bed.copy()
    output[:, :pair_count] = bed[:, :pair_count] - estimate
    return output


def pad_trim(samples: np.ndarray, count: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(samples) == count:
        return samples
    if len(samples) > count:
        return samples[:count].copy()
    out = np.zeros(count, dtype=np.float32)
    out[: len(samples)] = samples
    return out


def resample_linear(samples: np.ndarray, count: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(samples) == 0:
        return np.zeros(count, dtype=np.float32)
    if len(samples) == 1:
        return np.full(count, samples[0], dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    target_x = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def atempo_filter(speed: float) -> str:
    speed = float(np.clip(speed, 0.05, 20.0))
    parts: list[str] = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.5f}")
    return ",".join(parts)


def fit_duration(source: Path, destination: Path, duration: float, sample_rate: int = DIALOGUE_RATE) -> None:
    """Time-fit speech with ffmpeg atempo, then pad or trim to the exact length."""
    duration = max(0.05, float(duration))
    samples = load_mono(source, sample_rate)
    count = max(1, int(round(duration * sample_rate)))
    if len(samples) < 8:
        write_wav(destination, np.zeros(count, dtype=np.float32), sample_rate)
        return
    source_duration = len(samples) / sample_rate
    speed = source_duration / duration
    if 0.97 <= speed <= 1.03:
        fitted = pad_trim(samples, count)
    else:
        temporary = destination.with_suffix(".stretch.wav")
        stretched = destination.with_suffix(".stretched.wav")
        write_wav(temporary, samples, sample_rate)
        try:
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temporary),
                    "-filter:a",
                    atempo_filter(speed),
                    "-ar",
                    str(sample_rate),
                    str(stretched),
                ]
            )
            fitted = pad_trim(load_mono(stretched, sample_rate), count)
        finally:
            temporary.unlink(missing_ok=True)
            stretched.unlink(missing_ok=True)
    write_wav(destination, fitted, sample_rate)


def estimate_f0(samples: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 500.0) -> float:
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) < int(0.05 * sample_rate):
        return 160.0
    frame = samples[: int(0.45 * sample_rate)]
    frame = frame - float(np.mean(frame))
    if float(np.max(np.abs(frame))) < 1e-4:
        return 160.0
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(len(corr) - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 160.0
    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    if lag <= 0:
        return 160.0
    return float(sample_rate / lag)


def amplitude_envelope(samples: np.ndarray, sample_rate: int, win_s: float = 0.03, hop_s: float = 0.01) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    window = max(8, int(win_s * sample_rate))
    hop = max(4, int(hop_s * sample_rate))
    if len(samples) < window:
        value = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        return np.array([value], dtype=np.float32)
    values = []
    for start in range(0, len(samples) - window + 1, hop):
        frame = samples[start : start + window]
        values.append(float(np.sqrt(np.mean(frame * frame))))
    return np.asarray(values, dtype=np.float32)


def apply_envelope(samples: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) == 0 or len(envelope) == 0:
        return samples
    stretched = np.interp(
        np.linspace(0.0, 1.0, len(samples)),
        np.linspace(0.0, 1.0, len(envelope)),
        envelope,
    ).astype(np.float32)
    peak = float(np.max(stretched))
    if peak < 1e-6:
        return samples * 0.2
    stretched = stretched / peak
    stretched = 0.15 + 0.85 * stretched
    return samples * stretched


def pitch_shift(samples: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.25 or len(samples) < 16:
        return np.asarray(samples, dtype=np.float32)
    factor = 2.0 ** (semitones / 12.0)
    pitched = resample_linear(samples, max(8, int(round(len(samples) / factor))))
    return resample_linear(pitched, len(samples))


def slice_samples(samples: np.ndarray, sample_rate: int, start: float, end: float, pad: float = 0.0) -> np.ndarray:
    if end < start:
        start, end = end, start
    left = max(0, int(round((start - pad) * sample_rate)))
    right = min(len(samples), int(round((end + pad) * sample_rate)))
    if right <= left:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(samples[left:right], dtype=np.float32).copy()


def normalize_peak(samples: np.ndarray, peak: float = 0.9) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) == 0:
        return samples
    current = float(np.max(np.abs(samples)))
    if current < 1e-6:
        return samples
    return samples * (peak / current)


def line_fade_gain(length: int, sample_rate: int) -> np.ndarray:
    """Rise and fall inside one line. About 10 ms, never past the line edges."""
    gain = np.ones(length, dtype=np.float32)
    if length <= 0:
        return gain
    fade = int(round(LINE_FADE_SECONDS * sample_rate))
    fade = min(fade, length // 2)
    if fade <= 0:
        return gain
    gain[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    gain[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return gain


def mix_tracks(
    background: Path,
    lines: list[tuple[Path, float] | tuple[Path, float, float]],
    destination: Path,
    duration: float,
    vocals: Path | None = None,
) -> None:
    """Keep the original mix outside each dubbed line.

    Inside a line, fade onto the mix-minus-vocals bed and add the English
    dialogue. The fade is about 10 ms and stays inside the line. Channels
    past the stereo pair are copied through. Sounds that overlap speech can
    still be removed with the vocal estimate. Without a separate vocal file,
    the original mix stays under the new line.
    """
    original_pcm, sample_rate = _load_pcm(background)
    count = max(1, int(round(float(duration) * sample_rate)))
    original_pcm = _pad_trim_pcm(original_pcm, count)
    output_pcm = original_pcm.copy()
    original = original_pcm.astype(np.float32) / 32768.0
    pair = min(2, original.shape[1])
    estimate = None
    if vocals is not None and Path(vocals).resolve() != Path(background).resolve():
        estimate = _vocal_estimate(vocals, sample_rate, count, pair)
    removal = np.zeros(count, dtype=np.float32)
    added = np.zeros((count, pair), dtype=np.float32)
    for item in lines:
        path, start, end = _parse_line(item)
        mono, line_rate = _line_mono(path)
        if line_rate != sample_rate and len(mono):
            mono = resample_linear(mono, max(1, int(round(len(mono) * sample_rate / line_rate))))
        start_i = int(round(start * sample_rate))
        if start_i < 0:
            mono = mono[-start_i:]
            start_i = 0
        end_i = start_i + len(mono) if end is None else int(round(end * sample_rate))
        end_i = min(count, end_i)
        if start_i >= end_i or start_i >= count:
            continue
        length = end_i - start_i
        gain = line_fade_gain(length, sample_rate)
        dialogue = np.zeros(length, dtype=np.float32)
        take = min(length, len(mono))
        dialogue[:take] = mono[:take]
        removal[start_i:end_i] = np.maximum(removal[start_i:end_i], gain)
        added[start_i:end_i] += dialogue[:, None] * gain[:, None]
    mask = removal > 0
    if np.any(mask):
        stereo = original[:, :pair]
        mixed = stereo[mask]
        if estimate is not None:
            mixed = mixed - removal[mask, None] * estimate[mask]
        mixed = mixed + added[mask]
        output_pcm[mask, :pair] = _float_to_pcm(mixed)
    _write_pcm(destination, output_pcm, sample_rate)


def _parse_line(item: tuple[Path, float] | tuple[Path, float, float]) -> tuple[Path, float, float | None]:
    if len(item) == 2:
        path, start = item
        return path, float(start), None
    path, start, end = item
    return path, float(start), float(end)


def _line_mono(path: Path) -> tuple[np.ndarray, int]:
    pcm, sample_rate = _load_pcm(path)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32), sample_rate
    audio = pcm.astype(np.float32) / 32768.0
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return mono, sample_rate


def _vocal_estimate(path: Path, sample_rate: int, count: int, pair: int) -> np.ndarray:
    pcm, vocal_rate = _load_pcm(path)
    audio = pcm.astype(np.float32) / 32768.0
    if audio.ndim == 1:
        audio = audio[:, None]
    if vocal_rate != sample_rate and len(audio):
        target = max(1, int(round(len(audio) * sample_rate / vocal_rate)))
        audio = np.stack([resample_linear(audio[:, channel], target) for channel in range(audio.shape[1])], axis=1)
    audio = pad_trim_2d(audio, count)
    if audio.shape[1] == 1 and pair == 2:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :pair]


def _pad_trim_pcm(pcm: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, pcm.shape[1]), dtype="<i2")
    if pcm.shape[0] == count:
        return pcm.copy()
    if pcm.shape[0] > count:
        return pcm[:count].copy()
    out = np.zeros((count, pcm.shape[1]), dtype="<i2")
    out[: pcm.shape[0]] = pcm
    return out


def _float_to_pcm(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def _write_pcm(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    import wave

    audio = np.asarray(pcm, dtype="<i2")
    if audio.ndim == 1:
        audio = audio[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def mux(
    episode: Path,
    mix_wav: Path,
    destination: Path,
    *,
    lips: bool = False,
    lips_model_dir: Path | None = None,
    spoken_spans: list[tuple[float, float]] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".wav":
        audio_args = ["-c:a", "pcm_s16le"]
    elif suffix == ".flac":
        audio_args = ["-c:a", "flac"]
    else:
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
    video_input = episode
    video_out = has_video_stream(episode) and suffix not in {".wav", ".flac", ".mp3", ".ogg", ".aac"}
    if video_out and lips:
        video_input = _picture_for_lips(episode, mix_wav, destination, lips_model_dir, spoken_spans or [])
    if video_out:
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_input),
                "-i",
                str(mix_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                *audio_args,
                str(destination),
            ]
        )
        return
    if lips:
        print("[lips] there is no picture to patch", flush=True)
    run_cmd(["ffmpeg", "-y", "-i", str(mix_wav), *audio_args, str(destination)])


def _picture_for_lips(
    episode: Path,
    mix_wav: Path,
    destination: Path,
    model_dir: Path | None,
    spoken_spans: list[tuple[float, float]],
) -> Path:
    from anime_dub_tools.lips import lips_skip_message, render_lips

    message = lips_skip_message(model_dir)
    if message:
        print(message, flush=True)
        return episode
    patched = destination.with_name(f"{destination.stem}.lips-picture.mkv")
    try:
        render_lips(episode, mix_wav, patched, model_dir, spoken_spans)
    except Exception:
        patched.unlink(missing_ok=True)
        raise
    return patched
