from __future__ import annotations

from dataclasses import dataclass


class DubError(RuntimeError):
    """The pipeline cannot continue with the inputs or the selected backends."""


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str
    actor: str | None


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AssignedCue:
    cue: Cue
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerRow:
    speaker: str
    lines: int
    actor: str | None


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))
