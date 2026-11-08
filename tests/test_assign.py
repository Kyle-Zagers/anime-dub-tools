import numpy as np

from anime_dub_tools.assign import (
    acoustic_speakers,
    assign_cues,
    fit_window,
    pitch_ordered_voices,
    reference_intervals,
    speaker_rows,
)
from anime_dub_tools.models import AssignedCue, Cue, Turn


def test_actor_names_stay_consistent_when_diarization_flips():
    cues = [
        Cue(1, 0.0, 1.0, "Look out.", "Luna"),
        Cue(2, 1.2, 2.2, "I heard you.", "Sol"),
        Cue(3, 2.4, 3.4, "Move together.", "Luna"),
    ]
    turns = [
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(1.2, 2.2, "SPEAKER_01"),
        Turn(2.4, 3.4, "SPEAKER_01"),
    ]
    assigned = assign_cues(cues, turns)
    assert [item.speaker for item in assigned] == ["Luna", "Sol", "Luna"]
    rows = {row.speaker: row for row in speaker_rows(assigned)}
    assert rows["Luna"].lines == 2
    assert rows["Luna"].actor == "Luna"
    assert rows["Sol"].lines == 1
    intervals = reference_intervals("Luna", assigned, turns)
    assert intervals == [(0.0, 1.0), (2.4, 3.4)]
    assert reference_intervals("Sol", assigned, turns) == [(1.2, 2.2)]


def test_unnamed_lines_follow_diarized_speakers():
    cues = [
        Cue(1, 0.30, 1.25, "Look out.", None),
        Cue(2, 1.75, 2.70, "I heard you.", None),
        Cue(3, 3.25, 4.20, "Move together.", None),
    ]
    turns = [
        Turn(0.25, 1.35, "SPEAKER_00"),
        Turn(1.70, 2.80, "SPEAKER_01"),
        Turn(3.20, 4.30, "SPEAKER_00"),
    ]
    assigned = assign_cues(cues, turns)
    assert assigned[0].speaker == assigned[2].speaker == "SPEAKER_00"
    assert assigned[1].speaker == "SPEAKER_01"
    assert speaker_rows(assigned)[0].actor is None


def test_fit_window_keeps_the_subtitle_cue():
    cue = Cue(1, 0.0, 2.0, "A long caption.", None)
    turns = [Turn(0.40, 1.10, "SPEAKER_00")]
    start, end = fit_window(cue, turns)
    assert start == 0.0
    assert end == 2.0


def test_fit_window_does_not_crush_a_line_into_a_short_voiced_tail():
    cue = Cue(1, 10.0, 12.5, "A sentence that should fill the cue.", None)
    turns = [Turn(12.30, 12.48, "SPEAKER_00")]
    start, end = fit_window(cue, turns)
    assert (start, end) == (10.0, 12.5)
    assert end - start > 1.0


def test_unnamed_line_without_overlap_uses_the_nearest_speaker():
    cues = [Cue(1, 5.0, 5.4, "Late line.", None)]
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.2, 2.0, "SPEAKER_01")]
    assigned = assign_cues(cues, turns)
    assert assigned[0].speaker == "SPEAKER_01"


def test_acoustic_speakers_replace_a_script_id_when_the_center_overlaps():
    cues = [
        AssignedCue(Cue(1, 0.0, 1.0, "Look out.", None), "c1", 0.0, 1.0),
        AssignedCue(Cue(2, 1.2, 2.2, "I heard you.", None), "c1", 1.2, 2.2),
        AssignedCue(Cue(3, 8.0, 8.4, "Off mic.", None), "c2", 8.0, 8.4),
    ]
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.2, 2.2, "SPEAKER_01")]
    voiced = acoustic_speakers(cues, turns)
    assert [item.speaker for item in voiced] == ["SPEAKER_00", "SPEAKER_01", "c2"]
    assert voiced[0].line_text() == "Look out."


def test_lower_japanese_voice_takes_the_earlier_preset():
    rate = 16000
    low = np.sin(2 * np.pi * 120 * np.arange(rate) / rate).astype(np.float32)
    high = np.sin(2 * np.pi * 240 * np.arange(rate) / rate).astype(np.float32)
    audio = np.concatenate([low, high])
    assigned = [
        AssignedCue(Cue(1, 0.0, 1.0, "Low.", None), "SPEAKER_07", 0.0, 1.0),
        AssignedCue(Cue(2, 1.0, 2.0, "High.", None), "SPEAKER_00", 1.0, 2.0),
        AssignedCue(Cue(3, 0.0, 1.0, "Low again.", None), "SPEAKER_07", 0.0, 1.0),
    ]
    ordered = pitch_ordered_voices(assigned, audio, rate)
    assert [item.speaker for item in ordered] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
