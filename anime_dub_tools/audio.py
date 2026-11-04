from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from anime_dub_tools.models import DubError

DIALOGUE_RATE = 22050


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(episode),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


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
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


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


def mix_tracks(
    background: Path,
    lines: list[tuple[Path, float]],
    destination: Path,
    duration: float,
    sample_rate: int = DIALOGUE_RATE,
) -> None:
    count = max(1, int(round(duration * sample_rate)))
    mix = pad_trim(load_mono(background, sample_rate), count) * 0.8
    for path, start in lines:
        audio = load_mono(path, sample_rate)
        index = int(round(start * sample_rate))
        if index >= count:
            continue
        if index < 0:
            audio = audio[-index:]
            index = 0
        stop = min(count, index + len(audio))
        mix[index:stop] += audio[: stop - index]
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    write_wav(destination, mix, sample_rate)


def mux(episode: Path, mix_wav: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".wav":
        audio_args = ["-c:a", "pcm_s16le"]
    elif suffix == ".flac":
        audio_args = ["-c:a", "flac"]
    else:
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
    if has_video_stream(episode) and suffix not in {".wav", ".flac", ".mp3", ".ogg", ".aac"}:
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(episode),
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
    run_cmd(["ffmpeg", "-y", "-i", str(mix_wav), *audio_args, str(destination)])
