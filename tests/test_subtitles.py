from pathlib import Path

import pytest

from anime_dub_tools.models import DubError
from anime_dub_tools.subtitles import parse_subtitles

from tests.fixtures import ass_script, srt_script


def test_ass_uses_actor_names_and_strips_tags(tmp_path: Path):
    path = tmp_path / "episode.ass"
    path.write_text(ass_script(), encoding="utf-8")
    cues = parse_subtitles(path)
    assert [cue.actor for cue in cues] == ["ルナ", "Sol", "ルナ"]
    assert cues[0].text == "Look out for the signal."
    assert cues[0].start == pytest.approx(0.30)
    assert cues[0].end == pytest.approx(1.25)
    assert "an8" not in cues[0].text


def test_ass_accepts_an_actor_column(tmp_path: Path):
    path = tmp_path / "renamed.ass"
    path.write_text(
        "\n".join(
            [
                "[Events]",
                "Format: Start, End, Actor, Text",
                "Dialogue: 0:00:01.00,0:00:02.50,Pilot,Hold this heading.",
            ]
        ),
        encoding="utf-8",
    )
    cues = parse_subtitles(path)
    assert cues[0].actor == "Pilot"
    assert cues[0].text == "Hold this heading."
    assert cues[0].start == pytest.approx(1.0)
    assert cues[0].end == pytest.approx(2.5)


def test_ass_ignores_placeholder_names(tmp_path: Path):
    path = tmp_path / "blank.ass"
    path.write_text(
        "\n".join(
            [
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,Default,0,0,0,,Still a line.",
                "Dialogue: 0,0:00:01.00,0:00:02.00,Default,NTP,0,0,0,,Also unnamed.",
            ]
        ),
        encoding="utf-8",
    )
    cues = parse_subtitles(path)
    assert [cue.actor for cue in cues] == [None, None]


def test_srt_parses_without_actors(tmp_path: Path):
    path = tmp_path / "episode.srt"
    path.write_text(srt_script(), encoding="utf-8")
    cues = parse_subtitles(path)
    assert len(cues) == 3
    assert all(cue.actor is None for cue in cues)
    assert cues[1].text == "I heard you the first time."
    assert cues[1].start == pytest.approx(1.75)
    assert cues[2].end == pytest.approx(4.2)


def test_empty_and_unknown_subs_fail(tmp_path: Path):
    empty = tmp_path / "empty.srt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(DubError, match="No dialogue"):
        parse_subtitles(empty)
    odd = tmp_path / "notes.vtt"
    odd.write_text("WEBVTT\n", encoding="utf-8")
    with pytest.raises(DubError, match=".ass or .srt"):
        parse_subtitles(odd)
