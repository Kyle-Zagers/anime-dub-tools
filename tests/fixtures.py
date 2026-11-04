from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from anime_dub_tools.audio import write_wav


def write_two_speaker_episode(path: Path, sample_rate: int = 44100) -> float:
    """Two steady tones separated by silence. The first voice returns at the end."""
    duration = 4.6
    count = int(duration * sample_rate)
    audio = np.zeros(count, dtype=np.float32)
    _add_tone(audio, sample_rate, 0.25, 1.35, 180.0)
    _add_tone(audio, sample_rate, 1.70, 2.80, 420.0)
    _add_tone(audio, sample_rate, 3.20, 4.30, 180.0)
    write_wav(path, audio, sample_rate)
    return duration


def write_video(path: Path, wav: Path) -> None:
    commands = [
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=10:d=4.6",
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
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=10:d=4.6",
            "-i",
            str(wav),
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(path),
        ],
    ]
    errors = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and path.is_file():
            return
        errors.append(result.stderr[-500:])
    raise RuntimeError("Could not build a fixture video.\n" + "\n".join(errors))


def ass_script() -> str:
    return """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.30,0:00:01.25,Default,ルナ,0,0,0,,{\\an8}Look out for the signal.
Dialogue: 0,0:00:01.75,0:00:02.70,Default,Sol,0,0,0,,I heard you the first time.
Dialogue: 0,0:00:03.25,0:00:04.20,Default,ルナ,0,0,0,,Then we move together.
"""


def srt_script() -> str:
    return """1
00:00:00,300 --> 00:00:01,250
Look out for the signal.

2
00:00:01,750 --> 00:00:02,700
I heard you the first time.

3
00:00:03,250 --> 00:00:04,200
Then we move together.
"""


def _add_tone(audio: np.ndarray, sample_rate: int, start: float, end: float, frequency: float) -> None:
    left = int(start * sample_rate)
    right = int(end * sample_rate)
    time = np.arange(right - left, dtype=np.float32) / sample_rate
    audio[left:right] += 0.5 * np.sin(2.0 * np.pi * frequency * time)
