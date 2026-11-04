from pathlib import Path

import numpy as np

from anime_dub_tools.audio import (
    DIALOGUE_RATE,
    amplitude_envelope,
    apply_envelope,
    estimate_f0,
    fit_duration,
    load_mono,
    mix_tracks,
    write_wav,
)


def test_duration_fit_hits_shorter_and_longer_targets(tmp_path: Path):
    source = tmp_path / "source.wav"
    count = DIALOGUE_RATE
    time = np.arange(count, dtype=np.float32) / DIALOGUE_RATE
    write_wav(source, 0.4 * np.sin(2 * np.pi * 220 * time), DIALOGUE_RATE)
    for target in (0.22, 0.45, 1.7):
        destination = tmp_path / f"fit_{target}.wav"
        fit_duration(source, destination, target, DIALOGUE_RATE)
        audio = load_mono(destination, DIALOGUE_RATE)
        assert abs(len(audio) / DIALOGUE_RATE - target) < 0.02


def test_f0_and_envelope_follow_the_source():
    count = DIALOGUE_RATE
    time = np.arange(count, dtype=np.float32) / DIALOGUE_RATE
    low = 0.5 * np.sin(2 * np.pi * 150 * time)
    high = 0.5 * np.sin(2 * np.pi * 310 * time)
    assert abs(estimate_f0(low, DIALOGUE_RATE) - 150) < 8
    assert abs(estimate_f0(high, DIALOGUE_RATE) - 310) < 8
    shaped = low.copy()
    shaped[count // 2 :] *= 0.05
    envelope = amplitude_envelope(shaped, DIALOGUE_RATE)
    rendered = apply_envelope(np.ones(count, dtype=np.float32), envelope)
    first = float(np.sqrt(np.mean(rendered[: count // 2] ** 2)))
    second = float(np.sqrt(np.mean(rendered[count // 2 :] ** 2)))
    assert first > second * 1.5


def test_mix_places_a_line_on_a_silent_bed(tmp_path: Path):
    bed = tmp_path / "bed.wav"
    line = tmp_path / "line.wav"
    out = tmp_path / "mix.wav"
    write_wav(bed, np.zeros(DIALOGUE_RATE, dtype=np.float32), DIALOGUE_RATE)
    tone = 0.4 * np.sin(2 * np.pi * 440 * np.arange(DIALOGUE_RATE // 2, dtype=np.float32) / DIALOGUE_RATE)
    write_wav(line, tone, DIALOGUE_RATE)
    mix_tracks(bed, [(line, 0.25)], out, 1.0, DIALOGUE_RATE)
    mixed = load_mono(out, DIALOGUE_RATE)
    start = int(0.25 * DIALOGUE_RATE)
    assert float(np.max(np.abs(mixed[:start]))) < 1e-4
    assert float(np.max(np.abs(mixed[start : start + 1000]))) > 0.2
    assert float(np.max(np.abs(mixed))) <= 0.98
