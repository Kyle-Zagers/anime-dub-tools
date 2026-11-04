# Anime Dub Tools

Small Python scripts for preparing a video for dubbing: pull the audio and first subtitle track out with ffmpeg, then label who is speaking with pyannote.

## What it does

- `AnimeDubs.py` writes `extracted_audio.wav` and `extracted_subtitles.vtt` from a video file.
- `diarization.py` runs speaker diarization on that WAV, using a Japanese Callhome segmentation model, and writes `diarization_output.rttm`.
- `main.py` is a local launcher that calls the diarization script from a virtualenv.

## Setup

You need ffmpeg on your PATH, a CUDA-capable PyTorch install, and access to the gated pyannote model.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set HUGGINGFACE_API_KEY=your_token
python AnimeDubs.py path\to\video.mkv
python diarization.py
```

The Hugging Face token is read from the environment. Do not commit it.
