import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anime_dub_tools.audio import DIALOGUE_RATE, estimate_f0, has_video_stream, load_mono, stream_types
from anime_dub_tools.diarize import SpectralDiarizer
from anime_dub_tools.models import DubError
from anime_dub_tools.pipeline import dub_episode
from anime_dub_tools.separate import build_separator
from anime_dub_tools.synthesize import ReferenceSynthesizer

from tests.fixtures import ass_script, srt_script, write_two_speaker_episode, write_video


def test_pipeline_clones_pitch_from_the_episode_and_mixes(tmp_path: Path):
    episode = tmp_path / "episode.wav"
    subs = tmp_path / "episode.ass"
    out = tmp_path / "dubbed.mkv"
    work = tmp_path / "work"
    write_two_speaker_episode(episode)
    subs.write_text(ass_script(), encoding="utf-8")
    dub_episode(
        episode,
        subs,
        out,
        separator=build_separator("copy", None),
        diarizer=SpectralDiarizer(),
        synthesizer=ReferenceSynthesizer("harmonic"),
        work_dir=work,
    )
    assert out.is_file()
    assert "audio" in stream_types(out)
    assert "video" not in stream_types(out)
    refs = sorted(work.glob("ref_*.wav"))
    assert len(refs) == 2
    pitches = [estimate_f0(load_mono(path, DIALOGUE_RATE), DIALOGUE_RATE) for path in refs]
    assert max(pitches) - min(pitches) > 80
    line_one = load_mono(work / "line_001.wav", DIALOGUE_RATE)
    line_two = load_mono(work / "line_002.wav", DIALOGUE_RATE)
    assert abs(estimate_f0(line_one, DIALOGUE_RATE) - estimate_f0(line_two, DIALOGUE_RATE)) > 40
    assert float(np.sqrt(np.mean(line_one**2))) > 0.02


def test_cli_prints_the_speaker_map_and_fits_cues(tmp_path: Path):
    episode = tmp_path / "episode.wav"
    subs = tmp_path / "episode.ass"
    out = tmp_path / "dubbed.mkv"
    work = tmp_path / "work"
    write_two_speaker_episode(episode)
    subs.write_text(ass_script(), encoding="utf-8")
    result = _run(episode, subs, out, work)
    assert result.returncode == 0, result.stderr
    speakers = _line_speakers(result.stdout)
    assert speakers[1] == speakers[3] == "ルナ"
    assert speakers[2] == "Sol"
    assert "lines=2" in result.stdout
    assert "actor=ルナ" in result.stdout
    assert "lines=1" in result.stdout
    assert "actor=Sol" in result.stdout
    assert "backend=copy" in result.stdout
    assert "backend=spectral" in result.stdout
    start, end = _line_window(result.stdout, 1)
    audio = load_mono(work / "line_001.wav", DIALOGUE_RATE)
    assert abs(len(audio) / DIALOGUE_RATE - (end - start)) < 0.05
    assert "audio" in stream_types(out)


def test_cli_srt_and_video_mux(tmp_path: Path):
    tones = tmp_path / "tones.wav"
    video = tmp_path / "episode.mkv"
    subs = tmp_path / "episode.srt"
    out = tmp_path / "dubbed.mkv"
    work = tmp_path / "work"
    write_two_speaker_episode(tones)
    write_video(video, tones)
    subs.write_text(srt_script(), encoding="utf-8")
    result = _run(video, subs, out, work)
    assert result.returncode == 0, result.stderr
    speakers = _line_speakers(result.stdout)
    assert speakers[1] == speakers[3]
    assert speakers[1] != speakers[2]
    assert "actor=-" in result.stdout
    assert has_video_stream(out)
    assert "audio" in stream_types(out)
    mixed = load_mono(out, DIALOGUE_RATE)
    assert float(np.sqrt(np.mean(mixed**2))) > 0.01


def test_auto_separator_does_not_pretend_to_isolate():
    try:
        import audio_separator.separator  # noqa: F401
    except ImportError:
        with pytest.raises(DubError, match="audio-separator"):
            build_separator("auto", None)
    else:
        separator = build_separator("auto", None)
        assert separator.name == "roformer"


def test_missing_episode_is_an_error(tmp_path: Path):
    subs = tmp_path / "episode.srt"
    subs.write_text(srt_script(), encoding="utf-8")
    result = _run(tmp_path / "missing.mkv", subs, tmp_path / "out.mkv", tmp_path / "work")
    assert result.returncode == 1
    assert "Episode not found" in result.stderr


def _run(episode: Path, subs: Path, out: Path, work: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("HUGGINGFACE_API_KEY", None)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env.pop("INDEXTTS_MODEL_DIR", None)
    env.pop("OPENVOICE_DIR", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "anime_dub_tools",
            "--episode",
            str(episode),
            "--subs",
            str(subs),
            "--out",
            str(out),
            "--separator",
            "copy",
            "--diarizer",
            "spectral",
            "--tts",
            "reference",
            "--keep-work",
            str(work),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _line_speakers(stdout: str) -> dict[int, str]:
    found = {}
    for line in stdout.splitlines():
        if not line.startswith("LINE "):
            continue
        parts = line.split("\t")
        index = int(parts[0].split()[1])
        speaker = next(part.split("=", 1)[1] for part in parts if part.startswith("speaker="))
        found[index] = speaker
    return found


def _line_window(stdout: str, index: int) -> tuple[float, float]:
    for line in stdout.splitlines():
        if line.startswith(f"LINE {index}\t"):
            window = line.split("\t")[1]
            start, end = window.split("-")
            return float(start), float(end)
    raise AssertionError(f"LINE {index} missing from:\n{stdout}")
