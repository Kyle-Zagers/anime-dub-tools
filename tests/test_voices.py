import numpy as np

from anime_dub_tools.synthesize import build_synthesizer
from anime_dub_tools.voices import FEMALE_PRESETS, MALE_PRESETS, stable_voices


def _voices(vectors, starts, ends, pitches):
    return stable_voices(
        np.asarray(vectors, dtype=np.float64),
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
        pitches,
    )


def test_one_japanese_voice_stays_one_preset_across_scenes():
    vector = [1.0, 0.0]
    names = _voices(
        [vector, vector, vector, vector],
        [0.0, 1.0, 20.0, 21.0],
        [1.0, 2.0, 21.0, 22.0],
        [120.0, 130.0, 125.0, 280.0],
    )
    assert names == ["Ryan", "Ryan", "Ryan", "Ryan"]


def test_a_high_cluster_is_female_and_a_low_cluster_stays_male():
    low = [1.0, 0.0]
    high = [0.0, 1.0]
    names = _voices(
        [low, low, low, low, high, high, high, high],
        [i * 1.2 for i in range(8)],
        [i * 1.2 + 1.0 for i in range(8)],
        [120.0, 125.0, 118.0, 130.0, 230.0, 240.0, 235.0, 250.0],
    )
    assert names[:4] == ["Ryan", "Ryan", "Ryan", "Ryan"]
    assert names[4:] == ["Vivian", "Vivian", "Vivian", "Vivian"]


def test_the_next_line_keeps_its_voice_when_the_other_is_only_slightly_closer():
    own = np.array([1.0, 0.0])
    other = np.array([0.0, 1.0])
    borderline = np.array([0.60, 0.70])
    borderline = borderline / np.linalg.norm(borderline)
    to_own = 1.0 - float(borderline @ own)
    to_other = 1.0 - float(borderline @ other)
    assert to_other < to_own
    assert to_own <= to_other + 0.12
    names = _voices(
        [other, other, own, own, borderline],
        [0.0, 1.2, 2.4, 3.6, 4.8],
        [1.0, 2.2, 3.4, 4.6, 5.8],
        [120.0, 120.0, 122.0, 122.0, 121.0],
    )
    assert names[3] != names[0]
    assert names[4] == names[3]


def test_extra_male_clusters_share_a_male_preset():
    vectors = []
    starts = []
    ends = []
    pitches = []
    for speaker in range(6):
        vector = np.zeros(6)
        vector[speaker] = 1.0
        for copy in range(2):
            vectors.append(vector)
            start = (speaker * 2 + copy) * 1.2
            starts.append(start)
            ends.append(start + 1.0)
            pitches.append(110.0)
    names = _voices(vectors, starts, ends, pitches)
    assert set(names) <= set(MALE_PRESETS)
    assert not set(names) & set(FEMALE_PRESETS)
    assert len(set(names)) == 5


def test_auto_tts_uses_a_qwen_directory_without_extra_flags(tmp_path):
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text("{}", encoding="utf-8")
    synth = build_synthesizer("auto", None, None, "cpu", qwen_dir=qwen)
    assert synth.name == "qwen3"
    assert synth.model_dir == qwen
