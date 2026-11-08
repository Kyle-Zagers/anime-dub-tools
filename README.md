# Anime Dub Tools

Dub one anime episode into English. The input is the episode (video or audio) and its English subtitles. The output is the same episode with English dialogue mixed over the music and effects.

Speaker identity comes from the episode. A language model reads the English cues in order and gives each line a stable character id. The same person keeps that id for the whole episode, including lines that never say a name. Diarization can still cluster the audio, but a raw cluster label is not a character: one id is one voice, built from that character's own isolated dialogue. There is no voice library and no cast list.

The episode stays on your machine. This repository does not download, store, or ship any show.

## Command

```bash
export ANIME_DUB_SCRIPT_MODEL=/path/to/model.gguf

python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --script-model "$ANIME_DUB_SCRIPT_MODEL"
```

`EPISODE` can be a video or an audio file. `SUBS` is `.ass` or `.srt`. The words in the subtitle file are what gets spoken. Pass English subtitles. This tool does not translate.

The script model is required. `--script-model` or `ANIME_DUB_SCRIPT_MODEL` points at a GGUF file on this machine, so the subtitle text stays local. That file is read on CPU, and the GPU stays with separation, diarization, and the voice. An OpenAI-compatible server is optional and stays off until you set it:

```bash
export ANIME_DUB_SCRIPT_ENDPOINT=http://127.0.0.1:8080/v1
export ANIME_DUB_SCRIPT_MODEL=local-model

python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --script-endpoint "$ANIME_DUB_SCRIPT_ENDPOINT" \
  --script-model "$ANIME_DUB_SCRIPT_MODEL"
```

If neither is set, the command stops and names those variables. It does not fall back to espeak-ng, and it does not treat a diarizer label as a character. `ANIME_DUB_SCRIPT_API_KEY` is sent only when the endpoint is set, and it is not printed.

Vocal isolation has to be installed or this command stops. It will not quietly leave the original dialogue under the dub. A missing script model stops the command. Diarization falls back to the local clusterer when its token is missing, and the log names the backend that ran. The installs are in the next section.

A run prints a speaker map, one row per voice:

```text
SPEAKER MAP
c1	lines=12	actor=-
c2	lines=4	actor=-
```

`lines` is how many subtitle cues used that voice. The first column is the script character id. `actor` is the ASS name when every line of that character carried the same one, otherwise `-`.

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

`espeak-ng` is optional. It is an offline stub for wiring checks, not a finished dub. With it, `--tts reference` speaks the English words and then takes pitch and loudness from the episode. Without it, that same backend emits a tone with the speaker's pitch and the line's loudness.

## What a real episode needs

The default command isolates vocals, diarizes, and synthesizes. The quality of each stage depends on what is installed. Nothing below is bundled, and gated weights are not downloaded for you.

A real episode still needs a GPU, a Hugging Face token for pyannote, a local script model, and a local IndexTTS-2.5 checkpoint directory. MuseTalk weights are needed only when you pass `--lips`. Without them the picture is copied and the log says so. Missing IndexTTS, separator, or script-model weights stop with an error instead of a traceback.

| Stage | Backend | Install | Credentials |
| --- | --- | --- | --- |
| Isolate vocals | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) Mel-Band RoFormer | `pip install "audio-separator[gpu]"` or `pip install "audio-separator[cpu]"` | None. The checkpoint downloads on first use. |
| Diarize | [pyannote.audio](https://github.com/pyannote/pyannote-audio) `speaker-diarization-community-1` | `pip install "anime-dub-tools[diarize]"` | Hugging Face token. Accept the model conditions. GPU recommended, CPU works. |
| Script | A local GGUF through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), or an OpenAI-compatible endpoint you set | `pip install llama-cpp-python` and a GGUF file you already have | None for a local file. `ANIME_DUB_SCRIPT_API_KEY` only if you set an endpoint. |
| Speak | [IndexTTS-2.5](https://github.com/index-tts/index-tts) | `pip install git+https://github.com/index-tts/index-tts` and a local checkpoint directory | The checkpoint license is the upstream model license. GPU recommended. |
| Speak, MIT option | [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) + MeloTTS | See below | Checkpoints from the OpenVoice V2 Hugging Face repo. |

```bash
export HUGGINGFACE_API_KEY=hf_...
export INDEXTTS_MODEL_DIR=/path/to/indextts-checkpoints   # directory that contains config.yaml
export ANIME_DUB_SCRIPT_MODEL=/path/to/model.gguf

python -m anime_dub_tools \
  --episode "/path/to/episode.mkv" \
  --subs "/path/to/episode.en.ass" \
  --out dubbed.mkv \
  --separator roformer \
  --diarizer pyannote \
  --tts indextts \
  --tts-model-dir "$INDEXTTS_MODEL_DIR" \
  --script-model "$ANIME_DUB_SCRIPT_MODEL"
```

IndexTTS-2.5 stays the voice. It still clones a short reference, speaks English, takes emotion from a separate audio clip, and can aim at the cue with `duration_factor`. Checked again in November 2026, it is still the release that does all of that together, and it already runs on the largest GPU PyTorch can see (on a ROCm build that is the discrete card, not an NVIDIA wheel). Qwen3-TTS, Higgs Audio v3, CosyVoice 3, and Voxtral TTS 3 steer emotion with text or tags. OpenVoice only copies tone color. F5-TTS has no emotion channel. GPT-SoVITS wants a minute of finetune per speaker. Fish-style models treat the reference as timbre and put emotion in the text. None of those is a replacement. This is not a listening test.

The speaker reference is only the timbre prompt (`spk_audio_prompt`), one clip per script character, taken from that character's own isolated lines. Emotion is one style for the whole line, taken from that line's isolated source audio (`emo_audio_prompt`). Extreme takes raise `emo_alpha`. This is not a pitch-contour copy, and the text emotion model stays off. `duration_factor` stays between 0.5 and 2.0. ffmpeg `atempo` stays in that same range, then the line is padded or trimmed so it occupies the subtitle cue. Inference stays on one thread, on the selected GPU. While that line is inferred, other cores prepare the next line and time-fit the previous one. Subtitle parsing overlaps extraction, and the mix loads its files on a pool.

The vocal model is the Kimberley Jensen Mel-Band RoFormer bleedless checkpoint, `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt`, at 44.1 kHz. The vocal estimate is removed from the whole episode. The bed written out is stereo: the original pair minus that estimate. Channels past stereo are dropped, so source dialogue is not left on another output channel. English is added inside each cue. A fade of about 10 ms stays inside the line boundaries. Sounds that overlap the speech can still be removed with that vocal estimate. The bed is not the model's instrumental stem, and it is not ducked. A mono source is given a stereo pair for that subtraction. Pass `--separator-model` to try another checkpoint; a fullness-tuned fork is not the default.

OpenVoice V2 is the swappable MIT-licensed cloner. It copies tone color from the episode and gets English rhythm from MeloTTS, then the same time-fit runs. It does not follow the original performance as closely.

```bash
pip install git+https://github.com/myshell-ai/MeloTTS.git
pip install git+https://github.com/myshell-ai/OpenVoice.git
python -m unidic download
# Download V2 checkpoints so checkpoints_v2/converter/config.json exists.
export OPENVOICE_DIR=/path/to/checkpoints_v2

python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --tts openvoice --openvoice-dir "$OPENVOICE_DIR" \
  --script-model "$ANIME_DUB_SCRIPT_MODEL"
```

Swap the vocal model without code changes:

```bash
python -m anime_dub_tools --episode EPISODE --subs SUBS --out dubbed.mkv \
  --separator-model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

Any other filename audio-separator knows works the same way, including Demucs checkpoints it hosts. Spleeter is not used. Its vocal quality is well behind current RoFormer and Demucs models, and it pulls in an old TensorFlow stack.

pyannote `community-1` is language-agnostic and does not cap the cast at four speakers, which matters once an episode has a larger cast. It is gated. The older 3.1 pipeline and the Japanese Callhome segmentation swap are not used. Set `--diarizer-model` if you need a different pyannote pipeline later.

If `auto` cannot import audio-separator, the command stops and tells you how to install it. It will not pretend the original mix is a clean music bed. Diarization falls back to the local spectral clusterer when the token or package is missing, and the log says so. TTS does not. `--tts auto` with no IndexTTS checkpoint stops and tells you to set `--tts-model-dir`. `--tts reference` is the espeak-ng stub and is not a finished dub.

## What is implemented

| Piece | Status |
| --- | --- |
| ASS and SRT parsing, including the ASS name/actor column and override tags | Implemented. No translation. |
| Script | Implemented. A local GGUF, or an OpenAI-compatible endpoint you set, reads the cues and returns a character id per line. Missing config or weights stop with an error. The spoken words stay the subtitle line. |
| Speaker assignment | Implemented. The script id is the voice for the episode. Diarizer clusters only slice that character's own audio. |
| Vocal isolation | Implemented via audio-separator. Not runnable until that package is installed. `--separator copy` keeps the original mix underneath and says so. |
| Diarization | pyannote community-1 is implemented and swappable. Without a token, a NumPy spectral clusterer runs instead. It is a real clusterer, not production diarization. |
| Reference TTS | Implemented, and only when you pass `--tts reference`. espeak-ng speaks the line; pitch and loudness come from the speaker's audio and from that line. Harmonic fallback if espeak-ng is missing. This is a stub, not a finished dub. |
| IndexTTS and OpenVoice | Implemented against their current Python calls. They are not executed in tests. Weights are not downloaded here. |
| Time-fit | Implemented. The line occupies the subtitle cue. A short vocal blip inside a longer cue does not shrink it. ffmpeg `atempo` is clamped to 0.5–2.0, then the result is trimmed or padded. That fit runs on other cores while the next line is inferred, so it does not idle the GPU. IndexTTS `duration_factor` is clamped to 0.5–2.0 before that. |
| Mix and mux | Implemented. With a vocal estimate, the whole episode is a stereo bed (original pair minus that estimate) and channels past stereo are dropped. English is added inside each cue with a short fade. Loading the bed and the line wavs is multithreaded. Without a vocal estimate, the original mix is kept. Video inputs keep their picture when the output is a video container. |
| Lips | Off unless `--lips` is set. With MuseTalk weights, a mouth is patched only when one face is large, near-frontal, uncovered, and that face is the speaker. Every other frame, including off-screen lines, is copied. Missing weights copy the picture and say so. |

Signs, songs, and on-screen text are dubbed if they are dialogue lines in the subtitle file. Delete those cues first if you do not want them spoken.

## Local run without the large models

This checks the wiring only. The source dialogue is still in the bed, and the English voice is the espeak-ng stub, not a neural clone. The script model is still required.

```bash
python -m anime_dub_tools \
  --episode /path/to/episode.wav \
  --subs /path/to/episode.en.srt \
  --out dubbed.mkv \
  --separator copy \
  --diarizer spectral \
  --tts reference \
  --script-model "$ANIME_DUB_SCRIPT_MODEL" \
  --keep-work ./dub-work
```

## Tests

Tests build a few seconds of tones, a stereo-plus-extra-channel fixture, and a tiny ASS or SRT file. They cover parsing, speaker assignment, the mix-minus-vocals bed, emotion weights, the mouth gate, duration fitting, and mix/mux. They do not download weights and they do not need a GPU. `--lips` is checked with the weights missing, which must copy the picture.

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
  --tts-model-dir "$INDEXTTS_MODEL_DIR" \
  --script-model "$ANIME_DUB_SCRIPT_MODEL"
```

Any other episode uses the same command. The show title is not part of the tool.

## Flags

| Flag | Default | Role |
| --- | --- | --- |
| `--separator` | `auto` | `roformer` when audio-separator is installed, otherwise an error. `copy` keeps the original mix. |
| `--separator-model` | `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt` | Kimberley Jensen Mel-Band RoFormer bleedless checkpoint at 44.1 kHz. The vocal estimate is removed from the whole episode, and the bed is stereo. |
| `--diarizer` | `auto` | pyannote when the token and package are both present, otherwise spectral. |
| `--diarizer-model` | `pyannote/speaker-diarization-community-1` | Gated Hugging Face pipeline id. |
| `--tts` | `auto` | IndexTTS if `config.yaml` is in the model dir, else OpenVoice if its converter checkpoint is present. Missing weights stop. `reference` is the espeak-ng stub. |
| `--tts-model-dir` | `INDEXTTS_MODEL_DIR` | IndexTTS checkpoint directory. |
| `--openvoice-dir` | `OPENVOICE_DIR` | OpenVoice V2 `checkpoints_v2` directory. |
| `--script-model` | `ANIME_DUB_SCRIPT_MODEL` | Local GGUF path. When `--script-endpoint` is set, this is the model name that server expects. |
| `--script-endpoint` | `ANIME_DUB_SCRIPT_ENDPOINT` | Optional OpenAI-compatible base URL. Off unless set. |
| `--hf-token` | `HUGGINGFACE_API_KEY` or `HF_TOKEN` | Not printed. Do not commit it. |
| `--device` | `auto` | `cpu`, or the largest GPU PyTorch can see. Neural separation, diarization, and TTS use that device. An integrated GPU is not chosen when a larger one is present. |
| `--keep-work` | temporary directory | Leave extracted audio, references, and line wavs on disk. |
| `--lips` | off | Patch mouths with MuseTalk when one face passes the gate. Otherwise copy the frame. |
| `--lips-model-dir` | `MUSETALK_MODEL_DIR` | MuseTalk weights. Used only with `--lips`. Missing weights copy the picture. |
