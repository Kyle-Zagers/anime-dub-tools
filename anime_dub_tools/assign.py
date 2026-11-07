from __future__ import annotations

from anime_dub_tools.models import AssignedCue, Cue, SpeakerRow, Turn, overlap


def assign_cues(cues: list[Cue], turns: list[Turn]) -> list[AssignedCue]:
    """Give every line a stable speaker for this episode.

    A subtitle name/actor is the speaker id when the file has one, so the same
    name keeps one voice even if diarization flips a later line. Lines with no
    name follow the diarized speaker with the greatest overlap, which stays
    consistent for the episode because the diarizer labels are.
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
    """Vocal spans that represent this speaker.

    Named and unnamed speakers both prefer diarized speech that overlaps their
    lines. Cue windows are the fallback when diarization missed them.
    """
    own = [item for item in assigned if item.speaker == speaker]
    scores: dict[str, float] = {}
    for item in own:
        for turn in turns:
            covered = overlap(item.cue.start, item.cue.end, turn.start, turn.end)
            if covered > 0:
                scores[turn.speaker] = scores.get(turn.speaker, 0.0) + covered
    if scores:
        best = min(scores, key=lambda label: (-scores[label], label))
        return [(turn.start, turn.end) for turn in turns if turn.speaker == best]
    return [(item.cue.start, item.cue.end) for item in own]


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
        return best.start, best.end
    return cue.start, cue.end


def fit_window(cue: Cue, turns: list[Turn]) -> tuple[float, float]:
    """Place a line on the subtitle cue.

    A short diarized blip inside a longer cue is not the window. Using that
    blip starts the line late and then crushes the sentence into the tail.
    """
    del turns
    return cue.start, max(cue.start, cue.end)


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
