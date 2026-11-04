from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from anime_dub_tools.assign import (
    assign_cues,
    emotion_interval,
    format_line_log,
    format_speaker_map,
    reference_intervals,
    speaker_rows,
)
from anime_dub_tools.audio import (
    DIALOGUE_RATE,
    extract_audio,
    load_mono,
    media_duration,
    mix_tracks,
    mux,
    normalize_peak,
    require_ffmpeg,
    slice_samples,
    write_wav,
)
from anime_dub_tools.models import AssignedCue, DubError, Turn
from anime_dub_tools.subtitles import parse_subtitles


def dub_episode(
    episode: Path,
    subs: Path,
    out: Path,
    *,
    separator,
    diarizer,
    synthesizer,
    work_dir: Path,
) -> None:
    require_ffmpeg()
    if not episode.is_file():
        raise DubError(f"Episode not found: {episode}")
    work_dir.mkdir(parents=True, exist_ok=True)
    extracted = work_dir / "episode.wav"
    print(f"[extract] {episode}", flush=True)
    extract_audio(episode, extracted)
    duration = media_duration(extracted)
    if duration <= 0:
        raise DubError(f"No audio duration in {episode}")

    print(f"[separate] {separator.name}", flush=True)
    _emit_notice(separator)
    vocals, background = separator.separate(extracted, work_dir / "stems")

    print(f"[diarize] {diarizer.name}", flush=True)
    _emit_notice(diarizer)
    turns = diarizer.diarize(vocals)
    print(f"[diarize] turns={len(turns)} speakers={len({turn.speaker for turn in turns})}", flush=True)
    if not turns:
        print("[diarize] no speech turns found; unnamed lines share SPEAKER_00", flush=True)

    cues = parse_subtitles(subs)
    assigned = assign_cues(cues, turns)
    rows = speaker_rows(assigned)
    print(format_speaker_map(rows), flush=True)
    print(format_line_log(assigned), flush=True)

    vocal_audio = load_mono(vocals, DIALOGUE_RATE)
    references = _write_references(work_dir, vocal_audio, assigned, turns)
    print(f"[tts] {synthesizer.label}", flush=True)
    _emit_notice(synthesizer)
    rendered: list[tuple[Path, float]] = []
    for number, item in enumerate(assigned, start=1):
        start, end = _clamp(item, duration)
        if end - start < 0.05:
            print(f"[tts] skip line {item.cue.index}: cue is outside the episode", flush=True)
            continue
        emotion = _write_emotion(work_dir, vocal_audio, item, turns)
        destination = work_dir / f"line_{item.cue.index:03d}.wav"
        print(f"[tts] line {number}/{len(assigned)} speaker={item.speaker}", flush=True)
        try:
            synthesizer.synthesize(
                item.cue.text,
                references[item.speaker],
                emotion,
                end - start,
                destination,
            )
        except DubError:
            raise
        except Exception as exc:
            raise DubError(f"TTS failed on line {item.cue.index} ({item.speaker}): {exc}") from exc
        rendered.append((destination, start))

    if not rendered:
        raise DubError("Every subtitle line was skipped, so there is nothing to mix.")
    mix_path = work_dir / "mix.wav"
    print("[mix] dialogue onto background", flush=True)
    mix_tracks(background, rendered, mix_path, duration)
    print(f"[mux] {out}", flush=True)
    mux(episode, mix_path, out)
    print(f"wrote {out}", flush=True)


def _write_references(work_dir: Path, vocals: np.ndarray, assigned, turns: list[Turn]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in assigned:
        if item.speaker in paths:
            continue
        pieces = []
        budget = 12.0
        intervals = sorted(
            reference_intervals(item.speaker, assigned, turns),
            key=lambda span: span[1] - span[0],
            reverse=True,
        )
        for start, end in intervals:
            if budget <= 0.15:
                break
            clip = slice_samples(vocals, DIALOGUE_RATE, start, min(end, start + budget), pad=0.02)
            if len(clip) < int(0.12 * DIALOGUE_RATE):
                continue
            pieces.append(clip)
            budget -= len(clip) / DIALOGUE_RATE
        if not pieces:
            print(f"[tts] no isolated span for {item.speaker}; using the middle of the vocal stem", flush=True)
            mid = len(vocals) // 2
            pieces = [vocals[mid : mid + DIALOGUE_RATE]]
            if len(pieces[0]) < 8:
                pieces = [np.zeros(DIALOGUE_RATE // 2, dtype=np.float32)]
        gap = np.zeros(int(0.04 * DIALOGUE_RATE), dtype=np.float32)
        joined: list[np.ndarray] = []
        for index, piece in enumerate(pieces):
            if index:
                joined.append(gap)
            joined.append(piece)
        audio = normalize_peak(np.concatenate(joined), 0.9)
        path = work_dir / f"ref_{_stem(item.speaker)}.wav"
        write_wav(path, audio, DIALOGUE_RATE)
        paths[item.speaker] = path
    return paths


def _write_emotion(work_dir: Path, vocals: np.ndarray, item: AssignedCue, turns: list[Turn]) -> Path:
    start, end = emotion_interval(item.cue, turns)
    clip = slice_samples(vocals, DIALOGUE_RATE, start, end, pad=0.02)
    if len(clip) < int(0.12 * DIALOGUE_RATE):
        clip = slice_samples(vocals, DIALOGUE_RATE, item.start, item.end, pad=0.0)
    path = work_dir / f"emo_{item.cue.index:03d}.wav"
    if len(clip) < 8:
        clip = np.zeros(int(0.2 * DIALOGUE_RATE), dtype=np.float32)
    write_wav(path, normalize_peak(clip, 0.9), DIALOGUE_RATE)
    return path


def _clamp(item: AssignedCue, duration: float) -> tuple[float, float]:
    start = min(max(0.0, item.start), max(0.0, duration))
    end = min(max(start, item.end), duration)
    return start, end


def _stem(speaker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", speaker).strip("._")
    if cleaned:
        return cleaned[:60]
    digest = hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:8]
    return f"spk_{digest}"


def _emit_notice(stage) -> None:
    notice = getattr(stage, "notice", None)
    if notice:
        print(notice, flush=True)
