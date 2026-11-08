"""Script identity, faithful wording, and per-line emotion. Synthetic audio only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from anime_dub_tools.audio import DIALOGUE_RATE, load_mono, write_wav
from anime_dub_tools.models import Cue, DubError, Turn
from anime_dub_tools.assign import emotion_interval, reference_intervals
from anime_dub_tools.pipeline import dub_episode
from anime_dub_tools.script import (
    SYSTEM_PROMPT,
    EndpointScriptWriter,
    ScriptLine,
    apply_script,
    build_script_writer,
    faithful_spoken,
    light_cleanup,
    parse_script_response,
    script_from_complete,
    words,
)
from anime_dub_tools.separate import build_separator
from anime_dub_tools.synthesize import ReferenceSynthesizer, build_synthesizer

from tests.script_stub import ScriptStub

import pytest


def test_prompt_infers_identity_from_context_and_forbids_a_rewrite():
    text = SYSTEM_PROMPT.lower()
    assert "who answers" in text
    assert "no name" in text
    assert "do not paraphrase" in text
    assert "speaker_n" in text


def test_same_character_keeps_one_id_and_wording_stays_put():
    cues = [
        Cue(1, 0.0, 1.0, "Akira, wait.", None),
        Cue(2, 1.2, 2.0, "Where?", None),
        Cue(3, 3.0, 4.0, "Behind you.", None),
        Cue(4, 4.2, 5.0, "I see them.", None),
    ]
    turns = [
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(1.2, 2.0, "SPEAKER_01"),
        Turn(3.0, 4.0, "SPEAKER_04"),
        Turn(4.2, 5.0, "SPEAKER_01"),
    ]
    script = [
        ScriptLine(1, "c1", "Stop right now."),
        ScriptLine(2, "c2", "What place?"),
        ScriptLine(3, "c1", "Look behind."),
        ScriptLine(4, "c2", "They are visible."),
    ]
    assigned = apply_script(cues, script, turns)
    assert [item.speaker for item in assigned] == ["c1", "c2", "c1", "c2"]
    assert assigned[0].speaker == assigned[2].speaker
    assert assigned[1].speaker == assigned[3].speaker
    assert assigned[0].speaker != assigned[1].speaker
    assert assigned[1].line_text() == "Where?"
    assert assigned[3].line_text() == "I see them."
    assert assigned[0].line_text() == "Akira, wait."
    assert "Stop right now." not in assigned[0].line_text()
    assert reference_intervals("c1", assigned, turns) == [(0.0, 1.0), (3.0, 4.0)]
    assert reference_intervals("c2", assigned, turns) == [(1.2, 2.0), (4.2, 5.0)]


def test_a_merged_cluster_does_not_give_two_characters_one_voice():
    cues = [
        Cue(1, 0.0, 1.0, "Ken, move.", None),
        Cue(2, 1.2, 2.2, "I am already moving.", None),
    ]
    turns = [Turn(0.0, 2.2, "SPEAKER_00")]
    assigned = apply_script(
        cues,
        [
            ScriptLine(1, "c1", "Move along."),
            ScriptLine(2, "c2", "I have started."),
        ],
        turns,
    )
    assert [item.speaker for item in assigned] == ["c1", "c2"]
    assert reference_intervals("c1", assigned, turns) == [(0.0, 1.0)]
    assert reference_intervals("c2", assigned, turns) == [(1.2, 2.2)]
    assert assigned[0].line_text() == "Ken, move."


def test_emotion_stays_inside_this_line_when_a_cluster_runs_long():
    cue = Cue(1, 1.0, 2.0, "Wait.", None)
    assert emotion_interval(cue, [Turn(0.0, 5.0, "SPEAKER_00")]) == (1.0, 2.0)


def test_paraphrase_is_not_spoken_and_a_dash_is_only_markup():
    assert faithful_spoken("Look out for the signal.", "Watch for the sign.") == "Look out for the signal."
    assert words(faithful_spoken("- Where?", "What place?")) == ["where"]
    assert light_cleanup("- Where?") == "Where?"
    assert faithful_spoken("...", "He sighs.") == "..."


def test_a_diarizer_label_is_not_a_character():
    cues = [Cue(1, 0.0, 1.0, "Where?", None)]
    with pytest.raises(DubError, match="SPEAKER_01"):
        parse_script_response(json.dumps({"lines": [{"index": 1, "character": "SPEAKER_01", "text": "Where?"}]}), cues)


# The local model closed the lines array after the credit, then kept writing objects.
_EARLY_CLOSE = """{"lines": [{"index": 1, "character": null, "text": "Edited at https://subtitletools.com"}], 
{"index": 2, "character": "c1", "text": "Did you see that?"}, 
{"index": 3, "character": "c2", "text": "It's him!"}, 
{"index": 4, "character": "c2", "text": "That's the fabled..."}, 
{"index": 5, "character": "c2", "text": "Devil Z!"}, 
{"index": 6, "character": "c2", "text": "Devil Z"}, 
{"index": 7, "character": "c3", "text": "Several months earlier"}, 
{"index": 8, "character": "c1", "text": "You've gotta be kidding!"}, 
{"index": 9, "character": "c1", "text": "It's true! We're university students."}, 
{"index": 10, "character": "c3", "text": "Right, Akio?"}, 
{"index": 11, "character": "c3", "text": "What was it, again?"}, 
{"index": 12, "character": "c1", "text": "Don't blow our cover, idiot."}, 
{"index": 13, "character": "c1", "text": "Let's just give up on them and go home."}, 
{"index": 14, "character": "c2", "text": "So, what university are you guys at?"}, 
{"index": 15, "character": "c3", "text": "Oh, uh... Tokyo U, yeah."}, 
{"index": 16, "character": "c2", "text": "Hey..."}, 
{"index": 17, "character": "c1", "text": "Akio, what the hell was that?"}, 
{"index": 18, "character": "c3", "text": "It's the BlackBird!"}, 
{"index": 19, "character": "c2", "text": "You mean that BlackBird?"}, 
{"index": 20, "character": "c3", "text": "Let's go, Maa!"}}"""

_EARLY_CLOSE_CUES = [
    Cue(1, 0.0, 0.033, "Edited at https://subtitletools.com", None),
    Cue(2, 1.0, 2.0, "Did you see that?", None),
    Cue(3, 2.0, 3.0, "It's him!", None),
    Cue(4, 3.0, 4.0, "That's the fabled...", None),
    Cue(5, 4.0, 5.0, "Devil Z!", None),
    Cue(6, 5.0, 6.0, "Devil Z", None),
    Cue(7, 6.0, 7.0, "Several months earlier", None),
    Cue(8, 7.0, 8.0, "You've gotta be kidding!", None),
    Cue(9, 8.0, 9.0, "It's true! We're university students.", None),
    Cue(10, 9.0, 10.0, "Right, Akio?", None),
    Cue(11, 10.0, 11.0, "What was it, again?", None),
    Cue(12, 11.0, 12.0, "Don't blow our cover, idiot.", None),
    Cue(13, 12.0, 13.0, "Let's just give up on them and go home.", None),
    Cue(14, 13.0, 14.0, "So, what university are you guys at?", None),
    Cue(15, 14.0, 15.0, "Oh, uh... Tokyo U, yeah.", None),
    Cue(16, 15.0, 16.0, "Hey...", None),
    Cue(17, 16.0, 17.0, "Akio, what the hell was that?", None),
    Cue(18, 17.0, 18.0, "It's the BlackBird!", None),
    Cue(19, 18.0, 19.0, "You mean that BlackBird?", None),
    Cue(20, 19.0, 20.0, "Let's go, Maa!", None),
]


def test_early_array_close_skips_the_credit_and_keeps_the_later_objects():
    with pytest.raises(json.JSONDecodeError):
        json.loads(_EARLY_CLOSE)
    calls = {"n": 0}

    def complete(messages):
        del messages
        calls["n"] += 1
        return _EARLY_CLOSE

    lines = script_from_complete(complete, _EARLY_CLOSE_CUES, chunk_size=20)
    assert calls["n"] == 1
    assert [line.index for line in lines] == list(range(2, 21))
    assert [line.character for line in lines[:5]] == ["c1", "c2", "c2", "c2", "c2"]
    assert {line.character for line in lines} == {"c1", "c2", "c3"}
    assert all(not line.character.upper().startswith("SPEAKER_") for line in lines)
    assigned = apply_script(_EARLY_CLOSE_CUES, lines, [])
    assert [item.cue.index for item in assigned] == list(range(2, 21))
    assert all("subtitletools.com" not in item.line_text() for item in assigned)


def test_a_spoken_line_with_a_url_and_no_character_is_not_skipped():
    cues = [Cue(1, 0.0, 1.0, "Look at https://example.com now.", None)]
    payload = json.dumps({"lines": [{"index": 1, "character": None, "text": cues[0].text}]})
    with pytest.raises(DubError, match="no character id"):
        parse_script_response(payload, cues)


def test_a_bare_array_of_line_objects_is_a_valid_chunk():
    payload = (
        '[{"index": 21, "character": "c1", "text": "W–What?!"}, '
        '{"index": 22, "character": "c2", "text": "Woo-hoo! Go get him!"}]'
    )
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    cues = [
        Cue(21, 20.0, 21.0, "W–What?!", None),
        Cue(22, 21.0, 22.0, "Woo-hoo! Go get him!", None),
    ]
    calls = {"n": 0}

    def complete(messages):
        del messages
        calls["n"] += 1
        return payload

    lines = script_from_complete(complete, cues, chunk_size=2)
    assert calls["n"] == 1
    assert [(line.index, line.character) for line in lines] == [(21, "c1"), (22, "c2")]
    assert all(not line.character.upper().startswith("SPEAKER_") for line in lines)
    assigned = apply_script(cues, lines, [])
    assert [item.speaker for item in assigned] == ["c1", "c2"]
    assert [item.line_text() for item in assigned] == ["W–What?!", "Woo-hoo! Go get him!"]


def test_one_retry_then_stop_with_the_cue_range_and_raw_reply():
    cues = [Cue(1, 0.0, 1.0, "Where?", None), Cue(2, 1.2, 2.0, "Here.", None)]
    replies = ["this is not json", "still not json {{{"]

    def complete(messages):
        del messages
        return replies.pop(0)

    with pytest.raises(DubError, match="cues 1-2") as exc:
        script_from_complete(complete, cues, chunk_size=2)
    text = str(exc.value)
    assert "still not json" in text
    assert "espeak-ng" in text
    assert "SPEAKER_n" in text
    assert replies == []


def test_a_second_reply_is_accepted_and_a_diarizer_label_is_not_retried():
    cues = [Cue(1, 0.0, 1.0, "Where?", None)]
    replies = [
        "not json",
        json.dumps({"lines": [{"index": 1, "character": "c1", "text": "Where?"}]}),
    ]

    def complete(messages):
        del messages
        return replies.pop(0)

    lines = script_from_complete(complete, cues, chunk_size=1)
    assert [line.character for line in lines] == ["c1"]
    assert replies == []

    calls = {"n": 0}

    def labeled(messages):
        del messages
        calls["n"] += 1
        return json.dumps({"lines": [{"index": 1, "character": "SPEAKER_01", "text": "Where?"}]})

    with pytest.raises(DubError, match="SPEAKER_01"):
        script_from_complete(labeled, cues, chunk_size=1)
    assert calls["n"] == 1


def test_later_chunk_sees_the_roster():
    cues = [
        Cue(1, 0.0, 1.0, "Mira, look.", None),
        Cue(2, 2.0, 3.0, "I see it.", None),
    ]
    seen = []

    def complete(messages):
        user = json.loads(messages[-1]["content"])
        seen.append(user["roster"])
        cue = user["cues"][0]
        return json.dumps({"lines": [{"index": cue["index"], "character": "c1", "text": cue["text"]}]})

    lines = script_from_complete(complete, cues, chunk_size=1)
    assert [line.character for line in lines] == ["c1", "c1"]
    assert seen[0] == []
    assert seen[1][0]["character"] == "c1"
    assert "Mira, look." in seen[1][0]["example"]


def test_missing_script_model_fails_clearly(tmp_path: Path):
    with pytest.raises(DubError, match="ANIME_DUB_SCRIPT_MODEL") as missing:
        build_script_writer(None, None, None)
    text = str(missing.value)
    assert "ANIME_DUB_SCRIPT_ENDPOINT" in text
    assert "espeak-ng" in text
    assert "SPEAKER_n" in text
    with pytest.raises(DubError, match="does not download"):
        build_script_writer(str(tmp_path / "missing.gguf"), None, None)
    with pytest.raises(DubError, match="ANIME_DUB_SCRIPT_ENDPOINT"):
        build_script_writer("some-hosted-model", None, None)


def test_endpoint_error_does_not_keep_the_api_key():
    stub = ScriptStub(status=500, echo_authorization=True)
    key = "secret-token-value"
    try:
        writer = EndpointScriptWriter(stub.endpoint, "test-script", key)
        with pytest.raises(DubError, match="HTTP 500") as exc:
            writer.write([Cue(1, 0.0, 1.0, "Where?", None)])
    finally:
        stub.close()
    assert key not in str(exc.value)


def test_cli_without_a_script_model_names_the_setting(tmp_path: Path):
    episode = tmp_path / "episode.wav"
    subs = tmp_path / "episode.srt"
    write_wav(episode, np.zeros(1000, dtype=np.float32), 22050)
    subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nWhere?\n", encoding="utf-8")
    env = os.environ.copy()
    for name in (
        "ANIME_DUB_SCRIPT_MODEL",
        "ANIME_DUB_SCRIPT_ENDPOINT",
        "ANIME_DUB_SCRIPT_API_KEY",
        "INDEXTTS_MODEL_DIR",
        "OPENVOICE_DIR",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "anime_dub_tools",
            "--episode",
            str(episode),
            "--subs",
            str(subs),
            "--out",
            str(tmp_path / "out.mkv"),
            "--separator",
            "copy",
            "--diarizer",
            "spectral",
            "--tts",
            "reference",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    assert "ANIME_DUB_SCRIPT_MODEL" in result.stderr
    assert "ANIME_DUB_SCRIPT_ENDPOINT" in result.stderr
    assert "espeak-ng" in result.stderr
    assert "SPEAKER MAP" not in result.stdout


def test_auto_tts_without_weights_is_not_the_espeak_stub():
    with pytest.raises(DubError, match="not a finished dub"):
        build_synthesizer("auto", None, None, "cpu")
    assert "not a finished dub" in ReferenceSynthesizer("harmonic").notice
    assert "not a finished dub" in ReferenceSynthesizer("espeak").notice


def test_pipeline_gives_one_voice_per_character_and_emotion_per_line(tmp_path: Path):
    episode = tmp_path / "episode.wav"
    subs = tmp_path / "episode.srt"
    _write_episode(episode)
    subs.write_text(_subs(), encoding="utf-8")
    synth = _RecordingSynthesizer()

    class _Writer:
        def write(self, cues):
            ids = {1: "c1", 2: "c2", 3: "c1", 4: "c2"}
            return [ScriptLine(cue.index, ids[cue.index], "A different sentence entirely.") for cue in cues]

    class _Split:
        name = "split"

        def diarize(self, vocals):
            del vocals
            return [
                Turn(0.20, 1.00, "SPEAKER_00"),
                Turn(1.20, 2.00, "SPEAKER_09"),
                Turn(2.40, 3.20, "SPEAKER_03"),
                Turn(3.60, 4.40, "SPEAKER_09"),
            ]

    work = tmp_path / "work"
    dub_episode(
        episode,
        subs,
        tmp_path / "dubbed.mkv",
        separator=build_separator("copy", None),
        diarizer=_Split(),
        synthesizer=synth,
        script_writer=_Writer(),
        work_dir=work,
    )
    assert [call["text"] for call in synth.calls] == [
        "Akira, wait.",
        "Where?",
        "Behind you.",
        "I see them.",
    ]
    assert synth.calls[0]["speaker"] == synth.calls[2]["speaker"]
    assert synth.calls[1]["speaker"] == synth.calls[3]["speaker"]
    assert synth.calls[0]["speaker"] != synth.calls[1]["speaker"]
    assert "SPEAKER_" not in synth.calls[0]["speaker"].name
    assert synth.calls[0]["emotion"].name == "emo_001.wav"
    assert synth.calls[2]["emotion"].name == "emo_003.wav"
    assert synth.calls[0]["emotion"] != synth.calls[2]["emotion"]
    refs = sorted(path.name for path in work.glob("ref_*.wav"))
    assert refs == ["ref_c1.wav", "ref_c2.wav"]
    first = load_mono(work / "emo_001.wav", DIALOGUE_RATE)
    third = load_mono(work / "emo_003.wav", DIALOGUE_RATE)
    assert first.shape == third.shape
    assert not np.allclose(first, third)


class _RecordingSynthesizer:
    name = "record"
    label = "record"

    def __init__(self) -> None:
        self.calls = []

    def synthesize(self, text, speaker_reference, emotion_reference, target_duration, destination):
        self.calls.append(
            {
                "text": text,
                "speaker": speaker_reference,
                "emotion": emotion_reference,
            }
        )
        count = max(1, int(max(0.12, target_duration) * DIALOGUE_RATE))
        write_wav(destination, np.zeros(count, dtype=np.float32), DIALOGUE_RATE)


def _write_episode(path: Path) -> None:
    rate = 44100
    duration = 4.8
    audio = np.zeros(int(duration * rate), dtype=np.float32)
    _tone(audio, rate, 0.20, 1.00, 180.0, 0.15)
    _tone(audio, rate, 1.20, 2.00, 420.0, 0.45)
    _tone(audio, rate, 2.40, 3.20, 180.0, 0.20)
    burst_at = int(2.70 * rate)
    audio[burst_at : burst_at + 400] = 0.9
    _tone(audio, rate, 3.60, 4.40, 420.0, 0.45)
    write_wav(path, audio, rate)


def _tone(audio, rate, start, end, frequency, amplitude) -> None:
    left = int(start * rate)
    right = int(end * rate)
    time = np.arange(right - left, dtype=np.float32) / rate
    audio[left:right] += amplitude * np.sin(2.0 * np.pi * frequency * time)


def _subs() -> str:
    return """1
00:00:00,200 --> 00:00:01,000
Akira, wait.

2
00:00:01,200 --> 00:00:02,000
Where?

3
00:00:02,400 --> 00:00:03,200
Behind you.

4
00:00:03,600 --> 00:00:04,400
I see them.
"""
