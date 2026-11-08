from pathlib import Path

import numpy as np

from anime_dub_tools.audio import DIALOGUE_RATE, estimate_f0, load_mono, write_wav
from anime_dub_tools.diarize import SpectralDiarizer, build_diarizer, diarize_samples
from anime_dub_tools.models import DubError
from anime_dub_tools.synthesize import ReferenceSynthesizer, build_synthesizer, qwen_voice_for

from tests.fixtures import write_two_speaker_episode

import pytest


def test_spectral_diarizer_keeps_a_returning_voice(tmp_path: Path):
    episode = tmp_path / "tones.wav"
    write_two_speaker_episode(episode, 16000)
    turns = SpectralDiarizer().diarize(episode)
    assert len(turns) >= 3
    assert turns[0].speaker == turns[-1].speaker
    assert turns[1].speaker != turns[0].speaker
    assert turns[0].speaker.startswith("SPEAKER_")


def test_silence_has_no_speakers():
    assert diarize_samples(np.zeros(16000, dtype=np.float32), 16000) == []


def test_pyannote_without_a_token_is_refused():
    with pytest.raises(DubError, match="Hugging Face"):
        build_diarizer("pyannote", None, None, "cpu")


def test_reference_voices_follow_pitch_and_loudness(tmp_path: Path):
    sample_rate = DIALOGUE_RATE
    count = sample_rate
    time = np.arange(count, dtype=np.float32) / sample_rate
    low = tmp_path / "low.wav"
    high = tmp_path / "high.wav"
    emotion = tmp_path / "emotion.wav"
    write_wav(low, 0.5 * np.sin(2 * np.pi * 140 * time), sample_rate)
    write_wav(high, 0.5 * np.sin(2 * np.pi * 300 * time), sample_rate)
    shaped = 0.5 * np.sin(2 * np.pi * 140 * time)
    shaped[count // 2 :] *= 0.05
    write_wav(emotion, shaped, sample_rate)
    synth = ReferenceSynthesizer("harmonic")
    low_out = tmp_path / "low-line.wav"
    high_out = tmp_path / "high-line.wav"
    synth.synthesize("Wait for the next corner.", low, emotion, 1.0, low_out)
    synth.synthesize("Wait for the next corner.", high, high, 1.0, high_out)
    low_audio = load_mono(low_out, sample_rate)
    high_audio = load_mono(high_out, sample_rate)
    assert abs(len(low_audio) / sample_rate - 1.0) < 0.03
    assert abs(estimate_f0(low_audio, sample_rate) - 140) < 12
    assert abs(estimate_f0(high_audio, sample_rate) - 300) < 12
    first = float(np.sqrt(np.mean(low_audio[: count // 2] ** 2)))
    second = float(np.sqrt(np.mean(low_audio[count // 2 :] ** 2)))
    assert first > second * 1.4


def test_explicit_neural_tts_requires_its_checkpoints():
    with pytest.raises(DubError, match="tts-model-dir"):
        build_synthesizer("indextts", None, None, "cpu")
    with pytest.raises(DubError, match="tts-model-dir"):
        build_synthesizer("qwen3", None, None, "cpu")
    with pytest.raises(DubError, match="openvoice-dir"):
        build_synthesizer("openvoice", None, None, "cpu")


def test_qwen_voices_are_studio_presets_and_stay_put():
    assert qwen_voice_for("c1") == "Ryan"
    assert qwen_voice_for("c2") == "Aiden"
    assert qwen_voice_for("c3") == "Uncle_Fu"
    assert qwen_voice_for("c1") == qwen_voice_for("c1")
    assert len({qwen_voice_for("c1"), qwen_voice_for("c2"), qwen_voice_for("c3")}) == 3
