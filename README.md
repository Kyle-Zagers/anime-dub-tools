# Anime Dub Tools

Dub one anime episode into English. The input is the episode (video or audio) and its English subtitles. The output is the same episode with English dialogue mixed over the music and effects.

Speaker identity comes from the episode. When an ASS line has an Actor/Name, that name is the speaker for the whole episode. SRT files, and ASS lines with an empty name, use diarized speakers instead. Each voice is built from that speaker's own isolated dialogue. There is no voice library and no cast list.

The episode stays on your machine. This repository does not download, store, or ship any show.

## Command

```bash
python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv
```

`EPISODE` can be a video or an audio file. `SUBS` is `.ass` or `.srt`. The words in the subtitle file are what gets spoken. Pass English subtitles. This tool does not translate.

Vocal isolation has to be installed or this command stops. It will not quietly leave the original dialogue under the dub. Diarization and the speaking voice fall back to the local backends when their token or checkpoints are missing, and the log names the backend that ran. The installs are in the next section.

A run prints a speaker map, one row per voice:

```text
SPEAKER MAP
Luna	lines=12	actor=Luna
SPEAKER_01	lines=4	actor=-
```

`lines` is how many subtitle cues used that voice. `actor` is the ASS name when the file had one, otherwise `-`.

## Install

Python 3.10+, ffmpeg, and ffprobe are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Debian or Ubuntu:

```bash
sudo apt-get install -y ffmpeg espeak-ng
```

`espeak-ng` is optional. With it, the local TTS speaks English and then takes pitch and loudness from the episode. Without it, that same backend emits a tone with the speaker's pitch and the line's loudness, which is only useful for testing the mix.

## What a real episode needs

The default command isolates vocals, diarizes, and synthesizes. The quality of each stage depends on what is installed. Nothing below is bundled, and gated weights are not downloaded for you.

| Stage | Backend | Install | Credentials |
| --- | --- | --- | --- |
| Isolate vocals | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) Mel-Band RoFormer | `pip install "audio-separator[gpu]"` or `pip install "audio-separator[cpu]"` | None. The checkpoint downloads on first use. |
| Diarize | [pyannote.audio](https://github.com/pyannote/pyannote-audio) `speaker-diarization-community-1` | `pip install "anime-dub-tools[diarize]"` | Hugging Face token. Accept the model conditions. GPU recommended, CPU works. |
| Speak | [IndexTTS-2.5](https://github.com/index-tts/index-tts) | `pip install git+https://github.com/index-tts/index-tts` and a local checkpoint directory | The checkpoint license is the upstream model license. GPU recommended. |
| Speak, MIT option | [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) + MeloTTS | See below | Checkpoints from the OpenVoice V2 Hugging Face repo. |

```bash
export HUGGINGFACE_API_KEY=hf_...
export INDEXTTS_MODEL_DIR=/path/to/indextts-checkpoints   # directory that contains config.yaml

python -m anime_dub_tools \
  --episode "/path/to/episode.mkv" \
  --subs "/path/to/episode.en.ass" \
  --out dubbed.mkv \
  --separator roformer \
  --diarizer pyannote \
  --tts indextts \
  --tts-model-dir "$INDEXTTS_MODEL_DIR"
```

IndexTTS is the voice backend to use for a real dub. It clones each speaker from their isolated lines (`spk_audio_prompt`) and takes that line's own performance as the emotion reference (`emo_audio_prompt`). The text emotion model is left off. `duration_factor` aims at the cue, then ffmpeg time-fits whatever is left so the line does not drift.

OpenVoice V2 is the swappable MIT-licensed cloner. It copies tone color from the episode and gets English rhythm from MeloTTS, then the same time-fit runs. It does not follow the original performance as closely.

```bash
pip install git+https://github.com/myshell-ai/MeloTTS.git
pip install git+https://github.com/myshell-ai/OpenVoice.git
python -m unidic download
# Download V2 checkpoints so checkpoints_v2/converter/config.json exists.
export OPENVOICE_DIR=/path/to/checkpoints_v2

python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --tts openvoice --openvoice-dir "$OPENVOICE_DIR"
```

Swap the vocal model without code changes:

```bash
python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --separator-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

Any other filename audio-separator knows works the same way, including Demucs checkpoints it hosts. Spleeter is not used. Its vocal quality is well behind current RoFormer and Demucs models, and it pulls in an old TensorFlow stack.

pyannote `community-1` is language-agnostic and does not cap the cast at four speakers, which matters once an episode has a larger cast. It is gated. The older 3.1 pipeline and the Japanese Callhome segmentation swap are not used. Set `--diarizer-model` if you need a different pyannote pipeline later.

If `auto` cannot import audio-separator, the command stops and tells you how to install it. It will not pretend the original mix is a clean music bed. Diarization and TTS do not block the process when their token or checkpoints are missing: `auto` falls back to the local spectral diarizer and the local reference TTS, and the log says so.

## What is implemented

| Piece | Status |
| --- | --- |
| ASS and SRT parsing, including the ASS name/actor column and override tags | Implemented. No translation. |
| Speaker assignment | Implemented. Names win when present. Otherwise diarized labels, stable for the episode. |
| Vocal isolation | Implemented via audio-separator. Not runnable until that package is installed. `--separator copy` keeps the original mix underneath and says so. |
| Diarization | pyannote community-1 is implemented and swappable. Without a token, a NumPy spectral clusterer runs instead. It is a real clusterer, not production diarization. |
| Reference TTS | Implemented. espeak-ng speaks the line; pitch and loudness come from the speaker's audio and from that line. Harmonic fallback if espeak-ng is missing. |
| IndexTTS and OpenVoice | Implemented against their current Python calls. They are not executed in tests. Weights are not downloaded here. |
| Time-fit | Implemented with ffmpeg `atempo`, then a sample-accurate trim or pad onto the cue, or onto the vocal span inside that cue. |
| Mix and mux | Implemented. Dialogue is mixed onto the background. Video inputs keep their picture when the output is a video container. |

Signs, songs, and on-screen text are dubbed if they are dialogue lines in the subtitle file. Delete those cues first if you do not want them spoken.

## Local run without the large models

This checks the wiring only. The source dialogue is still in the bed, and the English voice is espeak-ng colored by the episode, not a neural clone.

```bash
python -m anime_dub_tools \
  --episode /path/to/episode.wav \
  --subs /path/to/episode.en.srt \
  --out dubbed.mkv \
  --separator copy \
  --diarizer spectral \
  --tts reference \
  --keep-work ./dub-work
```

## Tests

Tests build a few seconds of tones and a tiny ASS or SRT file. They cover parsing, speaker assignment, duration fitting, and mix/mux. They do not download weights.

```bash
pip install -e ".[dev]"
pytest
```

The optional import check stays off unless you ask for it. It imports the installed packages and still does not download gated weights:

```bash
ANIME_DUB_LIVE=1 pytest -m live_models
```

## First real episode

Supply the files yourself. One possible first run, after the installs above:

```bash
python -m anime_dub_tools \
  --episode "/path/to/Wangan Midnight - 01.mkv" \
  --subs "/path/to/Wangan Midnight - 01.en.ass" \
  --out "wangan-midnight-01-dubbed.mkv" \
  --separator roformer \
  --diarizer pyannote \
  --tts indextts \
  --tts-model-dir "$INDEXTTS_MODEL_DIR"
```

Any other episode uses the same command. The show title is not part of the tool.

## Flags

| Flag | Default | Role |
| --- | --- | --- |
| `--separator` | `auto` | `roformer` when audio-separator is installed, otherwise an error. `copy` keeps the original mix. |
| `--separator-model` | Mel-Band RoFormer checkpoint | Passed through to audio-separator. |
| `--diarizer` | `auto` | pyannote when the token and package are both present, otherwise spectral. |
| `--diarizer-model` | `pyannote/speaker-diarization-community-1` | Gated Hugging Face pipeline id. |
| `--tts` | `auto` | IndexTTS if `config.yaml` is in the model dir, else OpenVoice if its converter checkpoint is present, else reference. |
| `--tts-model-dir` | `INDEXTTS_MODEL_DIR` | IndexTTS checkpoint directory. |
| `--openvoice-dir` | `OPENVOICE_DIR` | OpenVoice V2 `checkpoints_v2` directory. |
| `--hf-token` | `HUGGINGFACE_API_KEY` or `HF_TOKEN` | Not printed. Do not commit it. |
| `--device` | `auto` | `cpu` or `cuda`. CUDA is used only by the neural backends. |
| `--keep-work` | temporary directory | Leave extracted audio, references, and line wavs on disk. |
