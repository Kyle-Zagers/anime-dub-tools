from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    fit_duration,
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
from anime_dub_tools.schedule import cpu_workers, overlap_line_work
from anime_dub_tools.subtitles import parse_subtitles
from anime_dub_tools.synthesize import emo_alpha_for_line


def dub_episode(
    episode: Path,
    subs: Path,
    out: Path,
    *,
    separator,
    diarizer,
    synthesizer,
    work_dir: Path,
    lips: bool = False,
    lips_model_dir: Path | None = None,
) -> None:
    require_ffmpeg()
    if not episode.is_file():
        raise DubError(f"Episode not found: {episode}")
    work_dir.mkdir(parents=True, exist_ok=True)
    extracted = work_dir / "episode.wav"
    print(f"[extract] {episode}", flush=True)
    with ThreadPoolExecutor(max_workers=cpu_workers(), thread_name_prefix="dub-cpu") as pool:
        cues_job = pool.submit(parse_subtitles, subs)
        extract_audio(episode, extracted)
        duration = media_duration(extracted)
        if duration <= 0:
            raise DubError(f"No audio duration in {episode}")

        print(f"[separate] {separator.name}", flush=True)
        _emit_notice(separator)
        vocals, background = separator.separate(extracted, work_dir / "stems")

        print(f"[diarize] {diarizer.name}", flush=True)
        _emit_notice(diarizer)
        vocal_audio_job = pool.submit(load_mono, vocals, DIALOGUE_RATE)
        turns = diarizer.diarize(vocals)
        print(f"[diarize] turns={len(turns)} speakers={len({turn.speaker for turn in turns})}", flush=True)
        if not turns:
            print("[diarize] no speech turns found; unnamed lines share SPEAKER_00", flush=True)

        cues = cues_job.result()
        assigned = assign_cues(cues, turns)
        rows = speaker_rows(assigned)
        print(format_speaker_map(rows), flush=True)
        print(format_line_log(assigned), flush=True)

        vocal_audio = vocal_audio_job.result()
        references = _write_references(work_dir, vocal_audio, assigned, turns, pool)
        print(f"[tts] {synthesizer.label}", flush=True)
        _emit_notice(synthesizer)
        print("[tts] next-line prep and time-fit overlap this line", flush=True)
        rendered = _render_lines(
            assigned, duration, work_dir, vocal_audio, turns, references, synthesizer, pool
        )

    if not rendered:
        raise DubError("Every subtitle line was skipped, so there is nothing to mix.")
    mix_path = work_dir / "mix.wav"
    vocal_bed = None if vocals.resolve() == background.resolve() else vocals
    if vocal_bed is None:
        print("[mix] original mix under the dub; no vocal estimate", flush=True)
    else:
        print("[mix] vocal-free stereo bed for the whole episode; English on each cue", flush=True)
    mix_tracks(
        background,
        [(path, start, end) for path, start, end in rendered],
        mix_path,
        duration,
        vocals=vocal_bed,
    )
    print(f"[mux] {out}", flush=True)
    mux(
        episode,
        mix_path,
        out,
        lips=lips,
        lips_model_dir=lips_model_dir,
        spoken_spans=[(start, end) for _path, start, end in rendered],
    )
    print(f"wrote {out}", flush=True)


@dataclass
class _LineJob:
    number: int
    total: int
    item: AssignedCue
    start: float
    end: float
    speaker_reference: Path
    destination: Path
    emotion: Path | None = None
    emo_alpha: float | None = None


def _render_lines(assigned, duration, work_dir, vocal_audio, turns, references, synthesizer, pool) -> list:
    jobs: list[_LineJob] = []
    for number, item in enumerate(assigned, start=1):
        start, end = _clamp(item, duration)
        if end - start < 0.05:
            print(f"[tts] skip line {item.cue.index}: cue is outside the episode", flush=True)
            continue
        jobs.append(
            _LineJob(
                number=number,
                total=len(assigned),
                item=item,
                start=start,
                end=end,
                speaker_reference=references[item.speaker],
                destination=work_dir / f"line_{item.cue.index:03d}.wav",
            )
        )
    if not jobs:
        return []
    split = callable(getattr(synthesizer, "infer_line", None))

    def prepare(job: _LineJob) -> _LineJob:
        job.emotion = _write_emotion(work_dir, vocal_audio, job.item, turns)
        if getattr(synthesizer, "name", None) == "indextts" and job.emotion is not None:
            job.emo_alpha = emo_alpha_for_line(load_mono(job.emotion, DIALOGUE_RATE))
        return job

    def infer(job: _LineJob):
        print(f"[tts] line {job.number}/{job.total} speaker={job.item.speaker}", flush=True)
        if not split:
            try:
                synthesizer.synthesize(
                    job.item.cue.text,
                    job.speaker_reference,
                    job.emotion,
                    job.end - job.start,
                    job.destination,
                )
            except DubError:
                raise
            except Exception as exc:
                raise DubError(
                    f"TTS failed on line {job.item.cue.index} ({job.item.speaker}): {exc}"
                ) from exc
            return None
        raw = job.destination.with_suffix(".tts.wav")
        try:
            if synthesizer.name == "indextts":
                if job.emotion is None or job.emo_alpha is None:
                    raise DubError(f"Line {job.item.cue.index} has no emotion prompt.")
                synthesizer.infer_line(
                    job.item.cue.text,
                    job.speaker_reference,
                    job.emotion,
                    job.end - job.start,
                    raw,
                    job.emo_alpha,
                )
            else:
                raw = synthesizer.infer_line(
                    job.item.cue.text,
                    job.speaker_reference,
                    job.emotion,
                    job.end - job.start,
                    job.destination,
                )
        except DubError:
            raw.unlink(missing_ok=True)
            raise
        except Exception as exc:
            raw.unlink(missing_ok=True)
            raise DubError(f"TTS failed on line {job.item.cue.index} ({job.item.speaker}): {exc}") from exc
        return raw

    def finish(job: _LineJob, _prepared: _LineJob, raw: Path | None):
        if raw is None:
            return job.destination, job.start, job.end
        try:
            fit_duration(raw, job.destination, job.end - job.start, DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)
        return job.destination, job.start, job.end

    return overlap_line_work(jobs, prepare, infer, finish, pool=pool)


def _write_references(
    work_dir: Path,
    vocals: np.ndarray,
    assigned,
    turns: list[Turn],
    pool: ThreadPoolExecutor,
) -> dict[str, Path]:
    speakers: list[str] = []
    for item in assigned:
        if item.speaker not in speakers:
            speakers.append(item.speaker)
    written = pool.map(lambda speaker: _write_one_reference(work_dir, vocals, assigned, turns, speaker), speakers)
    return dict(written)


def _write_one_reference(work_dir: Path, vocals: np.ndarray, assigned, turns: list[Turn], speaker: str) -> tuple[str, Path]:
    pieces = []
    budget = 12.0
    intervals = sorted(
        reference_intervals(speaker, assigned, turns),
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
        print(f"[tts] no isolated span for {speaker}; using the middle of the vocal stem", flush=True)
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
    path = work_dir / f"ref_{_stem(speaker)}.wav"
    write_wav(path, audio, DIALOGUE_RATE)
    return speaker, path


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
