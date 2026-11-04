import os
import sys

import spleeter
from pydub import AudioSegment
from pydub.playback import play


def extract_subtitles(input_file, audio_output, subtitle_output):
    # Extract a stereo WAV and the first subtitle track.
    os.system(
        f'ffmpeg -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_output}" '
        f'-map 0:s:0 -f webvtt "{subtitle_output}"'
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AnimeDubs.py <video-file>")
        raise SystemExit(1)
    extract_subtitles(sys.argv[1], "extracted_audio.wav", "extracted_subtitles.vtt")
