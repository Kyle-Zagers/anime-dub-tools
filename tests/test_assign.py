from anime_dub_tools.assign import assign_cues, fit_window, reference_intervals, speaker_rows
from anime_dub_tools.models import Cue, Turn


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
    assert intervals == [(0.0, 1.0)]


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


def test_fit_window_uses_the_vocal_span_inside_the_cue():
    cue = Cue(1, 0.0, 2.0, "A long caption.", None)
    turns = [Turn(0.40, 1.10, "SPEAKER_00")]
    start, end = fit_window(cue, turns)
    assert start == 0.40
    assert end == 1.10


def test_unnamed_line_without_overlap_uses_the_nearest_speaker():
    cues = [Cue(1, 5.0, 5.4, "Late line.", None)]
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.2, 2.0, "SPEAKER_01")]
    assigned = assign_cues(cues, turns)
    assert assigned[0].speaker == "SPEAKER_01"
