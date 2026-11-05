from __future__ import annotations

import re
from pathlib import Path

from anime_dub_tools.models import Cue, DubError

_ASS_TIME = re.compile(r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)[.](?P<cs>\d+)$")
_SRT_TIME = re.compile(
    r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)[,.](?P<ms>\d+)$"
)
_TAG = re.compile(r"\{[^}]*\}")
_IGNORED_ACTORS = {"", "default", "ntp", "*"}


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_subtitle_text(text: str) -> str:
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = _TAG.sub("", text)
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_actor(name: str | None) -> str | None:
    if name is None:
        return None
    cleaned = re.sub(r"\s+", " ", name).strip()
    if cleaned.lower() in _IGNORED_ACTORS:
        return None
    return cleaned


def parse_ass_timestamp(value: str) -> float:
    match = _ASS_TIME.match(value.strip())
    if not match:
        raise DubError(f"Could not parse an ASS timestamp: {value!r}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    fraction = match.group("cs")
    scale = 100 if len(fraction) <= 2 else 1000
    return hours * 3600 + minutes * 60 + seconds + int(fraction) / scale


def parse_srt_timestamp(value: str) -> float:
    match = _SRT_TIME.match(value.strip())
    if not match:
        raise DubError(f"Could not parse an SRT timestamp: {value!r}")
    fraction = match.group("ms")
    millis = int(fraction) * (10 ** (3 - len(fraction))) if len(fraction) < 3 else int(fraction[:3])
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + millis / 1000
    )


def parse_subtitles(path: Path) -> list[Cue]:
    if not path.is_file():
        raise DubError(f"Subtitle file not found: {path}")
    suffix = path.suffix.lower()
    text = read_text(path)
    if suffix == ".ass":
        cues = _parse_ass(text)
    elif suffix == ".srt":
        cues = _parse_srt(text)
    else:
        raise DubError(f"Subtitles must be .ass or .srt, got {path.name}")
    cues = [cue for cue in cues if cue.text]
    if not cues:
        raise DubError(f"No dialogue lines found in {path}")
    return cues


def _parse_ass(text: str) -> list[Cue]:
    columns = [
        "layer",
        "start",
        "end",
        "style",
        "name",
        "marginl",
        "marginr",
        "marginv",
        "effect",
        "text",
    ]
    cues: list[Cue] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        kind, _, payload = line.partition(":")
        kind = kind.strip().lower()
        if kind == "format":
            parsed = [part.strip().lower() for part in payload.split(",")]
            if "start" in parsed and "text" in parsed:
                columns = parsed
            continue
        if kind != "dialogue":
            continue
        parts = payload.strip().split(",", max(len(columns) - 1, 1))
        fields = {columns[i]: parts[i].strip() for i in range(min(len(columns), len(parts)))}
        if "text" not in fields or "start" not in fields or "end" not in fields:
            raise DubError("ASS Dialogue line is missing start, end, or text.")
        actor_key = next((key for key in ("name", "actor", "speaker") if key in fields), None)
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=parse_ass_timestamp(fields["start"]),
                end=parse_ass_timestamp(fields["end"]),
                text=clean_subtitle_text(fields["text"]),
                actor=normalize_actor(fields.get(actor_key) if actor_key else None),
            )
        )
    return cues


def _parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if not rows:
            continue
        try:
            time_index = next(i for i, row in enumerate(rows) if "-->" in row)
        except StopIteration:
            continue
        start_raw, _, end_raw = rows[time_index].partition("-->")
        end_raw = end_raw.split()[0] if end_raw.split() else end_raw
        body = clean_subtitle_text(" ".join(rows[time_index + 1 :]))
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=parse_srt_timestamp(start_raw),
                end=parse_srt_timestamp(end_raw),
                text=body,
                actor=None,
            )
        )
    return cues
