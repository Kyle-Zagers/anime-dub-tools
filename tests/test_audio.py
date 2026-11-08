from pathlib import Path

import numpy as np

from anime_dub_tools.audio import (
    DIALOGUE_ACTIVE_DB,
    DIALOGUE_RATE,
    active_rms,
    amplitude_envelope,
    apply_envelope,
    estimate_f0,
    fit_duration,
    load_mono,
    mix_tracks,
    prepare_dialogue_line,
    write_wav,
)


def test_fit_duration_keeps_a_normal_line_in_the_body(tmp_path: Path):
    source = tmp_path / "source.wav"
    count = int(1.2 * DIALOGUE_RATE)
    time = np.arange(count, dtype=np.float32) / DIALOGUE_RATE
    write_wav(source, 0.4 * np.sin(2 * np.pi * 220 * time), DIALOGUE_RATE)
    destination = tmp_path / "fit.wav"
    fit_duration(source, destination, 1.5, DIALOGUE_RATE)
    audio = load_mono(destination, DIALOGUE_RATE)
    assert abs(len(audio) / DIALOGUE_RATE - 1.5) < 0.02
    body = audio[: int(1.0 * DIALOGUE_RATE)]
    assert float(np.sqrt(np.mean(body**2))) > 0.05
    assert float(np.max(np.abs(audio[: int(0.2 * DIALOGUE_RATE)]))) > 0.05


def test_ordinary_line_is_not_sped_into_a_tenth_of_a_second(tmp_path: Path):
    source = tmp_path / "source.wav"
    count = DIALOGUE_RATE
    time = np.arange(count, dtype=np.float32) / DIALOGUE_RATE
    write_wav(source, 0.4 * np.sin(2 * np.pi * 220 * time), DIALOGUE_RATE)
    destination = tmp_path / "crushed.wav"
    fit_duration(source, destination, 0.12, DIALOGUE_RATE)
    audio = load_mono(destination, DIALOGUE_RATE)
    assert abs(len(audio) / DIALOGUE_RATE - 0.12) < 0.02
    window = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / DIALOGUE_RATE)
    peak = float(freqs[1 + int(np.argmax(spectrum[1:]))])
    assert peak < 600


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
    mix_tracks(bed, [(line, 0.25)], out, 1.0)
    mixed = load_mono(out, DIALOGUE_RATE)
    start = int(0.25 * DIALOGUE_RATE)
    assert float(np.max(np.abs(mixed[:start]))) < 1e-4
    body = mixed[start + 800 : start + 4000]
    peak = float(np.max(np.abs(body)))
    assert 0.05 < peak < 0.4
    assert float(np.max(np.abs(mixed))) <= 0.98


def test_loud_and_quiet_lines_land_on_one_level():
    rate = 44100
    time = np.arange(rate, dtype=np.float32) / rate
    loud = prepare_dialogue_line(0.55 * np.sin(2 * np.pi * 180 * time), rate)
    quiet = prepare_dialogue_line(0.02 * np.sin(2 * np.pi * 180 * time), rate)
    loud_db = 20 * np.log10(max(active_rms(loud), 1e-8))
    quiet_db = 20 * np.log10(max(active_rms(quiet), 1e-8))
    assert abs(loud_db - DIALOGUE_ACTIVE_DB) < 1.5
    assert abs(quiet_db - DIALOGUE_ACTIVE_DB) < 1.5
    assert abs(loud_db - quiet_db) < 1.0


def test_a_leveled_line_stays_dry():
    rate = 44100
    impulse = np.zeros(rate // 2, dtype=np.float32)
    impulse[0] = 0.8
    rendered = prepare_dialogue_line(impulse, rate)
    direct = float(np.max(np.abs(rendered[: int(0.008 * rate)])))
    tail = float(np.max(np.abs(rendered[int(0.012 * rate) : int(0.06 * rate)])))
    assert direct > 0.02
    assert tail < direct * 0.02
