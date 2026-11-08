from __future__ import annotations

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
