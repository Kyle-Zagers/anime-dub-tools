from __future__ import annotations

from dataclasses import replace

import numpy as np

from anime_dub_tools.audio import estimate_f0
from anime_dub_tools.models import AssignedCue, Cue, SpeakerRow, Turn, overlap


def assign_cues(cues: list[Cue], turns: list[Turn]) -> list[AssignedCue]:
    """Map cues to subtitle names or diarizer labels.

    The episode path does not use this. A script character id is the voice.
    A subtitle name still wins over a diarizer label when this helper is called
    on its own. Lines with no name follow the diarized speaker with the greatest
    overlap.
    """
    assigned: list[AssignedCue] = []
    for cue in cues:
        speaker = cue.actor if cue.actor else _speaker_for_cue(cue, turns)
        start, end = fit_window(cue, turns)
        assigned.append(AssignedCue(cue=cue, speaker=speaker, start=start, end=end))
    return assigned


def speaker_rows(assigned: list[AssignedCue]) -> list[SpeakerRow]:
    counts: dict[str, int] = {}
    actors: dict[str, set[str | None]] = {}
    for item in assigned:
        counts[item.speaker] = counts.get(item.speaker, 0) + 1
        actors.setdefault(item.speaker, set()).add(item.cue.actor)
    rows: list[SpeakerRow] = []
    for speaker, count in counts.items():
        names = {name for name in actors[speaker] if name}
        actor = next(iter(names)) if len(names) == 1 else None
        rows.append(SpeakerRow(speaker=speaker, lines=count, actor=actor))
    return sorted(rows, key=lambda row: row.speaker)


def format_speaker_map(rows: list[SpeakerRow]) -> str:
    lines = ["SPEAKER MAP"]
    for row in rows:
        actor = row.actor if row.actor else "-"
        lines.append(f"{row.speaker}\tlines={row.lines}\tactor={actor}")
    return "\n".join(lines)


def format_line_log(assigned: list[AssignedCue]) -> str:
    lines = []
    for item in assigned:
        actor = item.cue.actor or "-"
        lines.append(
            f"LINE {item.cue.index}\t{item.start:.3f}-{item.end:.3f}\t"
            f"speaker={item.speaker}\tactor={actor}"
        )
    return "\n".join(lines)


def reference_intervals(
    speaker: str, assigned: list[AssignedCue], turns: list[Turn]
) -> list[tuple[float, float]]:
    """Vocal spans from this character's own cues.

    A diarizer label is only a time hint. The whole cluster is not the voice:
    a cluster that also covers someone else is clipped to this character's
    cues, and a character split across clusters still contributes every cue.
    """
    own = [item for item in assigned if item.speaker == speaker]
    spans: list[tuple[float, float]] = []
    for item in own:
        hit = False
        for turn in turns:
            start = max(item.cue.start, turn.start)
            end = min(item.cue.end, turn.end)
            if end - start >= 0.05:
                spans.append((start, end))
                hit = True
        if not hit:
            spans.append((item.cue.start, item.cue.end))
    return spans


def emotion_interval(cue: Cue, turns: list[Turn]) -> tuple[float, float]:
    """The source performance that should shape this one line."""
    best: Turn | None = None
    best_overlap = 0.0
    for turn in turns:
        covered = overlap(cue.start, cue.end, turn.start, turn.end)
        if covered > best_overlap:
            best = turn
            best_overlap = covered
    if best is not None and best_overlap >= 0.12:
        start = max(best.start, cue.start)
        end = min(best.end, cue.end)
        if end > start:
            return start, end
    return cue.start, cue.end


def fit_window(cue: Cue, turns: list[Turn]) -> tuple[float, float]:
    """Place a line on the subtitle cue.

    A short diarized blip inside a longer cue is not the window. Using that
    blip starts the line late and then crushes the sentence into the tail.
    """
    del turns
    return cue.start, max(cue.start, cue.end)


def acoustic_speakers(assigned: list[AssignedCue], turns: list[Turn]) -> list[AssignedCue]:
    """Give each cue the voice that is actually talking on the dialogue bus.

    The script id is a guess from the English words. On a 5.1 episode the
    center channel is the performance, so the diarizer of that channel is
    the voice. A cue with no overlap keeps the script id.
    """
    if not turns:
        return assigned
    voiced: list[AssignedCue] = []
    for item in assigned:
        if _overlap_amount(item.cue, turns) <= 0:
            voiced.append(item)
            continue
        voiced.append(replace(item, speaker=_speaker_for_cue(item.cue, turns)))
    return voiced


def pitch_ordered_voices(
    assigned: list[AssignedCue], samples: np.ndarray, sample_rate: int
) -> list[AssignedCue]:
    """Give lower Japanese voices the earlier presets.

    Diarizer numbers are arbitrary. The first studio presets are the lower
    voices and the later presets are the higher ones, so a cluster's pitch
    picks the preset. Lines that share a cluster still share a preset.
    """
    if sample_rate <= 0 or not assigned:
        return assigned
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    pitches: dict[str, list[float]] = {}
    for item in assigned:
        found = _cue_pitch(audio, sample_rate, item.cue.start, item.cue.end)
        if found is None:
            continue
        pitches.setdefault(item.speaker, []).append(found)
    ranked = sorted(
        pitches,
        key=lambda speaker: (float(np.median(pitches[speaker])), speaker),
    )
    missing = sorted({item.speaker for item in assigned} - set(ranked))
    order = ranked + missing
    labels = {speaker: f"SPEAKER_{index:02d}" for index, speaker in enumerate(order)}
    return [replace(item, speaker=labels[item.speaker]) for item in assigned]


def _cue_pitch(samples: np.ndarray, sample_rate: int, start: float, end: float) -> float | None:
    left = int(start * sample_rate)
    right = max(left + 1, int(end * sample_rate))
    clip = samples[left:right]
    need = int(0.45 * sample_rate)
    if len(clip) < int(0.12 * sample_rate):
        return None
    if len(clip) > need:
        hop = max(1, int(0.05 * sample_rate))
        best_at = 0
        best = -1.0
        for index in range(0, len(clip) - need + 1, hop):
            energy = float(np.sqrt(np.mean(np.square(clip[index : index + need]))))
            if energy > best:
                best = energy
                best_at = index
        clip = clip[best_at : best_at + need]
    if float(np.sqrt(np.mean(np.square(clip)))) < 10 ** (-46 / 20):
        return None
    return estimate_f0(clip, sample_rate)


def _overlap_amount(cue: Cue, turns: list[Turn]) -> float:
    return max((overlap(cue.start, cue.end, turn.start, turn.end) for turn in turns), default=0.0)


def _speaker_for_cue(cue: Cue, turns: list[Turn]) -> str:
    if not turns:
        return "SPEAKER_00"
    best: Turn | None = None
    best_overlap = 0.0
    for turn in turns:
        covered = overlap(cue.start, cue.end, turn.start, turn.end)
        if covered > best_overlap or (
            covered == best_overlap and covered > 0 and best is not None and turn.speaker < best.speaker
        ):
            best = turn
            best_overlap = covered
    if best is not None and best_overlap > 0:
        return best.speaker
    midpoint = (cue.start + cue.end) / 2
    nearest = min(turns, key=lambda turn: abs(((turn.start + turn.end) / 2) - midpoint))
    return nearest.speaker
