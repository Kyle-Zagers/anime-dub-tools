"""Synthetic coverage for the stereo bed, per-line emotion, and optional lips."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from anime_dub_tools.audio import (
    BED_RATE,
    extract_audio,
    load_audio,
    mix_tracks,
    probe_audio,
    stereo_bed,
    write_audio,
    write_wav,
)
from anime_dub_tools.lips import (
    FaceGate,
    composite_mouth_only,
    lips_skip_message,
    render_gated_frames,
    select_patch,
)
from anime_dub_tools.models import DubError
from anime_dub_tools.separate import DEFAULT_SEPARATOR_MODEL, RoformerSeparator
from anime_dub_tools.synthesize import (
    BASE_EMO_ALPHA,
    EXTREME_EMO_ALPHA,
    IndexTTSSynthesizer,
    duration_factor_for,
    emo_alpha_for_line,
    indextts_infer_kwargs,
)

from tests.fixtures import srt_script, write_two_speaker_episode
from tests.test_pipeline import _run


def test_default_checkpoint_is_the_bleedless_roformer():
    assert DEFAULT_SEPARATOR_MODEL == "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"
    assert "fullness" not in DEFAULT_SEPARATOR_MODEL


def test_stereo_bed_subtracts_only_the_stereo_pair():
    mix = np.zeros((8, 4), dtype=np.float32)
    mix[:, 0] = 0.5
    mix[:, 1] = -0.25
    mix[:, 2] = 0.7
    mix[:, 3] = -0.4
    vocals = np.array([[0.2, -0.1]] * 8, dtype=np.float32)
    bed = stereo_bed(mix, vocals)
    assert bed.shape == (8, 4)
    np.testing.assert_allclose(bed[:, 0], 0.3)
    np.testing.assert_allclose(bed[:, 1], -0.15)
    np.testing.assert_allclose(bed[:, 2], 0.7)
    np.testing.assert_allclose(bed[:, 3], -0.4)
    assert not np.allclose(bed[:, 0], bed[:, 1])


def test_extract_keeps_channels_past_stereo(tmp_path: Path):
    source = tmp_path / "wide.wav"
    rate = 48000
    count = rate // 10
    audio = np.zeros((count, 4), dtype=np.float32)
    audio[:, 0] = 0.2
    audio[:, 1] = -0.2
    audio[:, 2] = 0.55
    audio[:, 3] = -0.55
    write_audio(source, audio, rate)
    extracted = tmp_path / "extracted.wav"
    extract_audio(source, extracted)
    sample_rate, channels = probe_audio(extracted)
    loaded, loaded_rate = load_audio(extracted)
    assert sample_rate == loaded_rate == BED_RATE
    assert channels == 4
    assert loaded.shape[1] == 4
    np.testing.assert_allclose(loaded[:, 0], 0.2, atol=0.01)
    np.testing.assert_allclose(loaded[:, 1], -0.2, atol=0.01)
    np.testing.assert_allclose(loaded[:, 2], 0.55, atol=0.01)
    np.testing.assert_allclose(loaded[:, 3], -0.55, atol=0.01)


def test_separator_builds_the_bed_from_the_mix_minus_vocals(tmp_path: Path):
    source = tmp_path / "mix.wav"
    count = BED_RATE // 5
    mix = np.zeros((count, 4), dtype=np.float32)
    mix[:, 0] = 0.5
    mix[:, 1] = -0.3
    mix[:, 2] = 0.8
    mix[:, 3] = 0.1
    write_audio(source, mix, BED_RATE)
    separator = RoformerSeparator(DEFAULT_SEPARATOR_MODEL)

    class _Fake:
        def separate(self, path: str):
            pair, rate = load_audio(path)
            vocals = np.zeros_like(pair)
            vocals[:, 0] = 0.2
            vocal_path = Path(path).with_name("model_vocals.wav")
            write_audio(vocal_path, vocals, rate)
            return [str(vocal_path)]

    separator._load = lambda work_dir: _Fake()  # type: ignore[method-assign]
    vocals, background = separator.separate(source, tmp_path / "work")
    bed, rate = load_audio(background)
    assert rate == BED_RATE
    assert bed.shape[1] == 4
    np.testing.assert_allclose(bed[:, 0], 0.3, atol=0.01)
    np.testing.assert_allclose(bed[:, 1], -0.3, atol=0.01)
    np.testing.assert_allclose(bed[:, 2], 0.8, atol=0.01)
    np.testing.assert_allclose(bed[:, 3], 0.1, atol=0.01)
    vocal_audio, _ = load_audio(vocals)
    assert vocal_audio.shape[1] == 2
    assert "instrumental" not in background.name


def test_missing_vocal_estimate_is_not_replaced_by_an_instrumental(tmp_path: Path):
    source = tmp_path / "mix.wav"
    write_audio(source, np.full((BED_RATE // 10, 2), 0.2, dtype=np.float32), BED_RATE)
    separator = RoformerSeparator(DEFAULT_SEPARATOR_MODEL)

    class _Fake:
        def separate(self, path: str):
            stem = Path(path).with_name("song_instrumental.wav")
            pair, rate = load_audio(path)
            write_audio(stem, pair, rate)
            return [str(stem)]

    separator._load = lambda work_dir: _Fake()  # type: ignore[method-assign]
    with pytest.raises(DubError, match="instrumental"):
        separator.separate(source, tmp_path / "work")


def test_mix_does_not_duck_or_fold_the_bed(tmp_path: Path):
    bed = tmp_path / "bed.wav"
    line = tmp_path / "line.wav"
    out = tmp_path / "mix.wav"
    count = BED_RATE
    audio = np.zeros((count, 3), dtype=np.float32)
    audio[:, 0] = 0.5
    audio[:, 1] = -0.2
    audio[:, 2] = 0.3
    write_audio(bed, audio, BED_RATE)
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(BED_RATE // 4, dtype=np.float32) / BED_RATE)
    write_wav(line, tone, BED_RATE)
    mix_tracks(bed, [(line, 0.25)], out, 1.0)
    mixed, rate = load_audio(out)
    assert rate == BED_RATE
    assert mixed.shape[1] == 3
    start = int(0.25 * BED_RATE)
    np.testing.assert_allclose(mixed[:start, 0], 0.5, atol=0.01)
    np.testing.assert_allclose(mixed[:start, 1], -0.2, atol=0.01)
    np.testing.assert_allclose(mixed[:, 2], 0.3, atol=0.01)
    window = mixed[start : start + 1000]
    assert float(np.max(np.abs(window[:, 0] - 0.5))) > 0.05
    assert float(np.max(np.abs(window[:, 1] + 0.2))) > 0.05


def test_duration_factor_stays_inside_the_indextts_range():
    assert duration_factor_for(0.05, "one two three four") == 0.5
    assert duration_factor_for(30.0, "one two") == 2.0
    middle = duration_factor_for(1.0, "one two three")
    assert 0.5 <= middle <= 2.0


def test_emo_alpha_rises_only_on_an_extreme_line():
    count = 22050
    time = np.arange(count, dtype=np.float32) / count
    calm = 0.4 * np.sin(2 * np.pi * 220 * time)
    burst = np.zeros(count, dtype=np.float32)
    burst[1000:1100] = 1.0
    assert emo_alpha_for_line(calm) == BASE_EMO_ALPHA == 0.6
    assert emo_alpha_for_line(burst) == EXTREME_EMO_ALPHA == 1.0
    assert emo_alpha_for_line(np.zeros(count, dtype=np.float32)) == BASE_EMO_ALPHA


def test_indextts_prompt_is_the_line_not_a_text_emotion(tmp_path: Path):
    speaker = tmp_path / "speaker.wav"
    emotion = tmp_path / "line.wav"
    destination = tmp_path / "out.wav"
    kwargs = indextts_infer_kwargs("Look out.", speaker, emotion, destination, 9.0, 1.0)
    assert kwargs["spk_audio_prompt"] == str(speaker)
    assert kwargs["emo_audio_prompt"] == str(emotion)
    assert kwargs["use_emo_text"] is False
    assert kwargs["emo_alpha"] == 1.0
    assert kwargs["duration_factor"] == 2.0
    with pytest.raises(DubError, match="emo_audio_prompt"):
        indextts_infer_kwargs("Look out.", speaker, None, destination, 1.0, 0.6)


def test_missing_indextts_weights_are_a_clear_error(tmp_path: Path):
    model = tmp_path / "indextts"
    model.mkdir()
    (model / "config.yaml").write_text("model: missing\n", encoding="utf-8")
    speaker = tmp_path / "speaker.wav"
    emotion = tmp_path / "emotion.wav"
    write_wav(speaker, np.zeros(1000, dtype=np.float32))
    write_wav(emotion, np.linspace(0.0, 0.2, 1000, dtype=np.float32))
    synthesizer = IndexTTSSynthesizer(model, "cpu")
    with pytest.raises(DubError, match="IndexTTS-2.5"):
        synthesizer.synthesize("Look out.", speaker, emotion, 0.8, tmp_path / "line.wav")


def _face(**overrides) -> FaceGate:
    values = {
        "bbox": (0.2, 0.1, 0.4, 0.6),
        "yaw_degrees": 4.0,
        "mouth_visible": True,
        "mouth_covered": False,
        "is_speaker": True,
    }
    values.update(overrides)
    return FaceGate(**values)


def test_mouth_is_patched_only_for_one_clear_speaker():
    good = _face()
    assert select_patch([good], True) == good
    assert select_patch([good, good], True) is None
    assert select_patch([_face(bbox=(0.4, 0.4, 0.2, 0.1))], True) is None
    assert select_patch([_face(yaw_degrees=40.0)], True) is None
    assert select_patch([_face(mouth_visible=False)], True) is None
    assert select_patch([_face(mouth_covered=True)], True) is None
    assert select_patch([_face(is_speaker=False)], True) is None
    assert select_patch([good], False) is None
    assert select_patch([], True) is None


def test_composite_changes_only_the_mouth_box():
    original = np.full((30, 40, 3), (1, 2, 3), dtype=np.uint8)
    edited = np.full_like(original, 9)
    box = (5, 20, 8, 4)
    output = composite_mouth_only(original, edited, box)
    assert np.array_equal(output[20:24, 5:13], edited[20:24, 5:13])
    untouched = np.array(output, copy=True)
    untouched[20:24, 5:13] = original[20:24, 5:13]
    assert np.array_equal(untouched, original)


def test_offscreen_and_crowded_frames_are_copied():
    frame = np.full((20, 20, 3), 3, dtype=np.uint8)
    speaker = _face(bbox=(0.1, 0.1, 0.8, 0.8))

    def paint(source, box):
        del box
        painted = np.array(source, copy=True)
        painted[:] = 7
        return painted

    rendered = render_gated_frames(
        [frame, frame, frame],
        [0.2, 0.2, 3.0],
        [[speaker], [speaker, speaker], []],
        [(0.0, 1.0)],
        paint,
    )
    assert not np.array_equal(rendered[0], frame)
    assert np.array_equal(rendered[0][:10], frame[:10])
    assert np.array_equal(rendered[1], frame)
    assert np.array_equal(rendered[2], frame)


def test_missing_musetalk_weights_name_the_gap(tmp_path: Path):
    assert lips_skip_message(None) is not None
    message = lips_skip_message(tmp_path / "empty")
    assert message is not None
    assert "copied the picture unchanged" in message
    assert "MuseTalk" in message


def test_lips_flag_without_weights_copies_the_picture(tmp_path: Path):
    tones = tmp_path / "tones.wav"
    video = tmp_path / "episode.mkv"
    subs = tmp_path / "episode.srt"
    out = tmp_path / "dubbed.mkv"
    write_two_speaker_episode(tones)
    _write_pattern_video(video, tones)
    subs.write_text(srt_script(), encoding="utf-8")
    result = _run(video, subs, out, tmp_path / "work", extra=["--lips"])
    assert result.returncode == 0, result.stderr
    assert "copied the picture unchanged" in result.stdout
    assert "patched_frames" not in result.stdout
    assert np.array_equal(_frame(video), _frame(out))


def _write_pattern_video(path: Path, wav: Path) -> None:
    frames = []
    for index in range(12):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + index * 12
        frame[:, :, 1] = np.arange(32, dtype=np.uint8) * 4
        frame[:, :, 2] = (np.arange(24, dtype=np.uint8) * 6)[:, None]
        frames.append(frame)
    raw = path.with_suffix(".rgb")
    raw.write_bytes(b"".join(frame.tobytes() for frame in frames))
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "32x24",
            "-r",
            "10",
            "-i",
            str(raw),
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    raw.unlink(missing_ok=True)
    if result.returncode != 0 or not path.is_file():
        raise RuntimeError(result.stderr[-500:])


def _frame(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-500:])
    return np.frombuffer(result.stdout, dtype=np.uint8).copy()
