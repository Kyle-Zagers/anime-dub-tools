from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from anime_dub_tools.audio import (
    DIALOGUE_RATE,
    amplitude_envelope,
    apply_envelope,
    estimate_f0,
    fit_duration,
    place_natural,
    load_mono,
    normalize_peak,
    pitch_shift,
    write_wav,
)
from anime_dub_tools.models import DubError


def speech_seconds(text: str) -> float:
    words = max(1, len(text.split()))
    return max(0.35, words / 2.6)


BASE_EMO_ALPHA = 0.6
EXTREME_EMO_ALPHA = 1.0


def duration_factor_for(target_duration: float, text: str) -> float:
    return float(np.clip(target_duration / speech_seconds(text), 0.5, 2.0))


def emo_alpha_for_line(samples: np.ndarray) -> float:
    """One style weight for the whole line, from how extreme the take is.

    This is not a pitch contour. A steady take stays at the base weight.
    A shouted or bursty take raises it. The same value covers the whole line.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return BASE_EMO_ALPHA
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms < 1e-4:
        return BASE_EMO_ALPHA
    crest = float(np.max(np.abs(audio))) / rms
    hop = 256
    usable = audio[: len(audio) - (len(audio) % hop)]
    if usable.size < hop * 4:
        modulation = 0.0
    else:
        energy = np.sqrt(np.mean(usable.reshape(-1, hop) ** 2, axis=1))
        modulation = float(np.std(energy) / (float(np.mean(energy)) + 1e-6))
    if crest >= 4.5 or modulation >= 0.85:
        return EXTREME_EMO_ALPHA
    return BASE_EMO_ALPHA


def indextts_infer_kwargs(
    text: str,
    speaker: Path,
    emotion: Path | None,
    destination: Path,
    factor: float,
    emo_alpha: float,
) -> dict:
    if emotion is None:
        raise DubError(
            "IndexTTS-2.5 needs the line's isolated audio as emo_audio_prompt. "
            "The speaker reference is only the timbre prompt. No text emotion model is used."
        )
    bounded = float(np.clip(factor, 0.5, 2.0))
    return {
        "spk_audio_prompt": str(speaker),
        "text": text,
        "output_path": str(destination),
        "emo_audio_prompt": str(emotion),
        "emo_alpha": float(emo_alpha),
        "use_emo_text": False,
        "verbose": False,
        "interval_silence": 80,
        "lang": "EN",
        "duration_factor": bounded,
    }


class ReferenceSynthesizer:
    """English pronunciation from espeak-ng, identity from the episode audio.

    Pitch is shifted toward the speaker reference. Loudness follows the
    per-line source performance. No emotion classifier and no external voice
    library. Without espeak-ng, a harmonic tone still carries that pitch and
    loudness so the rest of the pipeline can be tested; it is not English speech.
    """

    def __init__(self, engine: str = "auto") -> None:
        if engine not in {"auto", "espeak", "harmonic"}:
            raise DubError(f"Unknown reference engine {engine!r}.")
        self.engine = engine

    @property
    def name(self) -> str:
        return "reference"

    @property
    def label(self) -> str:
        if self.engine == "harmonic" or (self.engine == "auto" and not espeak_available()):
            return "reference/harmonic"
        return "reference/espeak-ng"

    @property
    def notice(self) -> str:
        if self.label.endswith("harmonic"):
            return (
                "backend=reference/harmonic. This is an offline stub, not a finished dub. "
                "espeak-ng is not available, so each line is a tone at the speaker's pitch "
                "with the line's loudness contour. It is not English speech."
            )
        return (
            "backend=reference/espeak-ng. This is an offline stub, not a finished dub. "
            "espeak-ng speaks the English words. Pitch and loudness come from this episode. "
            "It is not a cloned voice. A real dub uses --tts indextts."
        )

    def synthesize(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> None:
        target_duration = max(0.12, float(target_duration))
        speaker = load_mono(speaker_reference, DIALOGUE_RATE)
        target_f0 = estimate_f0(speaker, DIALOGUE_RATE)
        emotion_path = emotion_reference or speaker_reference
        envelope = amplitude_envelope(load_mono(emotion_path, DIALOGUE_RATE), DIALOGUE_RATE)
        raw = destination.with_suffix(".raw.wav")
        shifted = destination.with_suffix(".pitch.wav")
        try:
            self._base(text, target_f0, target_duration, raw)
            fit_duration(raw, shifted, target_duration, DIALOGUE_RATE)
            audio = load_mono(shifted, DIALOGUE_RATE)
            current_f0 = estimate_f0(audio, DIALOGUE_RATE)
            if current_f0 > 0 and target_f0 > 0:
                semitones = 12.0 * np.log2(target_f0 / current_f0)
                audio = pitch_shift(audio, DIALOGUE_RATE, float(semitones))
            audio = apply_envelope(audio, envelope)
            audio = normalize_peak(audio, 0.85)
            write_wav(destination, audio, DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)
            shifted.unlink(missing_ok=True)

    def _base(self, text: str, target_f0: float, target_duration: float, destination: Path) -> None:
        use_espeak = self.engine == "espeak" or (self.engine == "auto" and espeak_available())
        if use_espeak:
            try:
                _espeak(text, target_f0, target_duration, destination)
                return
            except DubError:
                if self.engine == "espeak":
                    raise
        count = max(1, int(round(target_duration * DIALOGUE_RATE)))
        write_wav(destination, harmonic_stack(target_f0, count, DIALOGUE_RATE), DIALOGUE_RATE)


def _nvidia_cuda(device: str) -> bool:
    """bfloat16 stays on NVIDIA. Other CUDA-API devices, including ROCm, stay fp32."""
    try:
        import torch

        index = int(device.split(":")[-1]) if ":" in device else 0
        return "nvidia" in torch.cuda.get_device_name(index).lower()
    except Exception:
        return False


class IndexTTSSynthesizer:
    """IndexTTS-2.5 cloning. Timbre and emotion are separate prompts.

    The speaker reference is only ``spk_audio_prompt``. ``emo_audio_prompt`` is
    that line's isolated source audio, one style for the whole line. Extreme
    lines get a higher ``emo_alpha``. ``duration_factor`` stays in 0.5–2.0 and
    ffmpeg atempo finishes the fit. The text emotion model stays off.
    """

    name = "indextts"

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self._tts = None
        self.notice = (
            "backend=indextts. Timbre comes from the speaker reference. "
            "Emotion is one style taken from that line's isolated audio, "
            "with a higher emo_alpha when the take is extreme. Not a pitch contour."
        )

    @property
    def label(self) -> str:
        return "indextts"

    def synthesize(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> None:
        if emotion_reference is None:
            raise DubError(
                "IndexTTS-2.5 needs the line's isolated audio as emo_audio_prompt."
            )
        alpha = emo_alpha_for_line(load_mono(emotion_reference, DIALOGUE_RATE))
        raw = destination.with_suffix(".tts.wav")
        try:
            self.infer_line(text, speaker_reference, emotion_reference, target_duration, raw, alpha)
            fit_duration(raw, destination, max(0.12, float(target_duration)), DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)

    def infer_line(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path,
        target_duration: float,
        raw_path: Path,
        emo_alpha: float,
    ) -> None:
        """GPU inference only. ffmpeg time-fit stays with the caller."""
        tts = self._load()
        target_duration = max(0.12, float(target_duration))
        print(f"[tts] emo_alpha={emo_alpha:.2f} prompt=line-audio", flush=True)
        factor = duration_factor_for(target_duration, text)
        try:
            self._infer(tts, text, speaker_reference, emotion_reference, raw_path, factor, emo_alpha)
            produced = _wav_seconds(raw_path)
            if produced > 0 and not 0.85 <= produced / target_duration <= 1.15:
                adjusted = float(np.clip(factor * (target_duration / produced), 0.5, 2.0))
                self._infer(
                    tts, text, speaker_reference, emotion_reference, raw_path, adjusted, emo_alpha
                )
        except DubError:
            raise
        except Exception as exc:
            raise DubError(
                "IndexTTS-2.5 failed while cloning a line. Check --tts-model-dir and the GPU. "
                f"{exc}"
            ) from exc

    def _load(self):
        if self._tts is not None:
            return self._tts
        config = self.model_dir / "config.yaml"
        if not config.is_file():
            raise DubError(
                f"IndexTTS-2.5 config.yaml was not found in {self.model_dir}. "
                "Download the 2.5 checkpoints into that directory and pass --tts-model-dir."
            )
        try:
            from indextts.infer_v2_5 import IndexTTS2
        except ImportError as exc:
            raise DubError(
                "IndexTTS-2.5 is not installed. Per-line emotion uses that release only.\n"
                "  pip install git+https://github.com/index-tts/index-tts\n"
                "Put config.yaml and the 2.5 weights in --tts-model-dir. "
                "This pipeline does not download them."
            ) from exc
        try:
            self._tts = IndexTTS2(
                cfg_path=str(config),
                model_dir=str(self.model_dir),
                device=self.device,
                use_bf16=self.device.startswith("cuda") and _nvidia_cuda(self.device),
                use_qwen_emo=False,
            )
        except Exception as exc:
            raise DubError(
                f"Could not load IndexTTS-2.5 weights from {self.model_dir}. "
                f"The checkpoint files are missing or unreadable. {exc}"
            ) from exc
        return self._tts

    @staticmethod
    def _infer(
        tts,
        text: str,
        speaker: Path,
        emotion: Path | None,
        destination: Path,
        factor: float,
        emo_alpha: float,
    ) -> None:
        tts.infer(**indextts_infer_kwargs(text, speaker, emotion, destination, factor, emo_alpha))
        if not destination.is_file():
            raise DubError(f"IndexTTS-2.5 did not write {destination}")


class OpenVoiceSynthesizer:
    """MeloTTS English plus OpenVoice V2 tone-color conversion.

    The reference clip supplies timbre. Rhythm comes from the base English
    speaker, then ffmpeg places the line on the cue. MIT-licensed alternative
    to IndexTTS when tone color is enough.
    """

    name = "openvoice"

    def __init__(self, checkpoint_dir: Path, device: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._ready = None
        self._target_cache: dict[str, object] = {}
        self.notice = (
            "backend=openvoice. Tone color is cloned from each speaker's isolated lines. "
            "English rhythm comes from MeloTTS, then the line is time-fit to the cue. "
            "OpenVoice watermarks conversions with its public @MyShell tag."
        )

    @property
    def label(self) -> str:
        return "openvoice"

    def synthesize(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> None:
        colored = self.infer_line(
            text, speaker_reference, emotion_reference, target_duration, destination
        )
        try:
            fit_duration(colored, destination, max(0.12, float(target_duration)), DIALOGUE_RATE)
        finally:
            colored.unlink(missing_ok=True)

    def infer_line(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> Path:
        """GPU clone only. Returns the unfitted wav. ffmpeg time-fit stays with the caller."""
        del emotion_reference
        converter, tts, speaker_id, source_se = self._load()
        target_duration = max(0.12, float(target_duration))
        speed = float(np.clip(speech_seconds(text) / target_duration, 0.6, 1.8))
        key = str(speaker_reference.resolve())
        if key not in self._target_cache:
            from openvoice import se_extractor

            target_se, _name = se_extractor.get_se(str(speaker_reference), converter, vad=True)
            self._target_cache[key] = target_se
        raw = destination.with_suffix(".base.wav")
        colored = destination.with_suffix(".clone.wav")
        try:
            tts.tts_to_file(text, speaker_id, str(raw), speed=speed)
            converter.convert(
                audio_src_path=str(raw),
                src_se=source_se,
                tgt_se=self._target_cache[key],
                output_path=str(colored),
                message="@MyShell",
            )
            if not colored.is_file():
                raise DubError(f"OpenVoice did not write {colored}")
            return colored
        except Exception:
            colored.unlink(missing_ok=True)
            raise
        finally:
            raw.unlink(missing_ok=True)

    def _load(self):
        if self._ready is not None:
            return self._ready
        converter_dir = self.checkpoint_dir / "converter"
        config = converter_dir / "config.json"
        checkpoint = converter_dir / "checkpoint.pth"
        if not config.is_file() or not checkpoint.is_file():
            raise DubError(
                "OpenVoice V2 checkpoints were not found at "
                f"{converter_dir}. Download them from "
                "https://huggingface.co/myshell-ai/OpenVoiceV2 and pass --openvoice-dir "
                "pointing at the checkpoints_v2 directory."
            )
        try:
            import torch
            from melo.api import TTS
            from openvoice.api import ToneColorConverter
        except ImportError as exc:
            raise DubError(
                "OpenVoice V2 is not installed.\n"
                "  pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
                "  pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
                "  python -m unidic download"
            ) from exc
        converter = ToneColorConverter(str(config), device=self.device)
        converter.load_ckpt(str(checkpoint))
        tts = None
        language = "EN_NEWEST"
        try:
            tts = TTS(language=language, device=self.device)
        except Exception:
            language = "EN"
            tts = TTS(language=language, device=self.device)
        speaker_ids = dict(tts.hps.data.spk2id)
        if not speaker_ids:
            raise DubError(f"MeloTTS language {language} did not expose a speaker id.")
        speaker_key = _prefer_english_key(list(speaker_ids))
        speaker_id = speaker_ids[speaker_key]
        se_path = _source_embedding(self.checkpoint_dir, speaker_key)
        try:
            source_se = torch.load(se_path, map_location=self.device, weights_only=True)
        except TypeError:
            source_se = torch.load(se_path, map_location=self.device)
        except Exception:
            source_se = torch.load(se_path, map_location=self.device, weights_only=False)
        self._ready = (converter, tts, speaker_id, source_se)
        return self._ready


# Studio voices shipped with Qwen3-TTS CustomVoice. English-native speakers
# come first so the busiest characters are not asked to perform in a dialect
# voice. These are not cloned from the episode.
QWEN_STUDIO_VOICES = (
    "Ryan",
    "Aiden",
    "Uncle_Fu",
    "Eric",
    "Dylan",
    "Vivian",
    "Serena",
    "Ono_Anna",
    "Sohee",
)


def qwen_voice_for(character: str) -> str:
    """A clean preset voice for one character id.

    The isolated episode clip is not a prompt. Cloning it copies the
    separation noise into every line. A diarizer label uses the same list,
    in the order the labels are numbered.
    """
    text = character.strip()
    numbered = re.fullmatch(r"(?:SPEAKER_)?(\d+)", text, flags=re.IGNORECASE)
    if numbered:
        index = int(numbered.group(1))
        if text[0] in {"c", "C"}:
            index -= 1
        if index >= 0:
            return QWEN_STUDIO_VOICES[index % len(QWEN_STUDIO_VOICES)]
    if len(text) > 1 and text[0] in {"c", "C"} and text[1:].isdigit():
        index = int(text[1:]) - 1
        if index >= 0:
            return QWEN_STUDIO_VOICES[index % len(QWEN_STUDIO_VOICES)]
    digest = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
    return QWEN_STUDIO_VOICES[digest % len(QWEN_STUDIO_VOICES)]


def character_from_reference(path: Path) -> str:
    stem = path.stem
    if stem.startswith("ref_"):
        return stem[4:]
    return stem


class QwenSynthesizer:
    """Qwen3-TTS CustomVoice. Preset studio voices, not a clone of the episode.

    IndexTTS was copying the noise in its training data and in the separated
    anime reference. CustomVoice speaks with the model's own clean voices.
    One character id keeps one preset for the whole episode.
    """

    name = "qwen3"
    natural_speed = True

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self._model = None
        self._seen: set[str] = set()
        self.notice = (
            "backend=qwen3. Qwen3-TTS CustomVoice uses its studio presets. "
            "The isolated episode audio is not the voice prompt."
        )

    @property
    def label(self) -> str:
        return "qwen3"

    def synthesize(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> None:
        raw = self.infer_line(text, speaker_reference, emotion_reference, target_duration, destination)
        try:
            place_natural(raw, destination, DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)

    def infer_line(
        self,
        text: str,
        speaker_reference: Path,
        emotion_reference: Path | None,
        target_duration: float,
        destination: Path,
    ) -> Path:
        """GPU speech only. Returns the unfitted wav. ffmpeg time-fit stays with the caller."""
        del emotion_reference, target_duration
        model = self._load()
        character = character_from_reference(speaker_reference)
        voice = qwen_voice_for(character)
        if character not in self._seen:
            self._seen.add(character)
            print(f"[tts] {character}={voice}", flush=True)
        raw = destination.with_suffix(".qwen.wav")
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language="English",
                speaker=voice,
            )
            audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            write_wav(raw, audio, int(sample_rate))
        except DubError:
            raw.unlink(missing_ok=True)
            raise
        except Exception as exc:
            raw.unlink(missing_ok=True)
            raise DubError(
                "Qwen3-TTS failed while speaking a line. Check --tts-model-dir and the GPU. "
                f"{exc}"
            ) from exc
        if not raw.is_file():
            raise DubError(f"Qwen3-TTS did not write {raw}")
        return raw

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.model_dir.is_dir():
            raise DubError(
                f"Qwen3-TTS weights were not found in {self.model_dir}. "
                "Download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice into that directory "
                "and pass --tts-model-dir. This pipeline does not download them."
            )
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise DubError(
                "Qwen3-TTS is not installed.\n"
                "  pip install qwen-tts\n"
                "Flash attention is not required. SDPA is used so an AMD GPU can run it."
            ) from exc
        dtype = torch.bfloat16 if _nvidia_cuda(self.device) else torch.float32
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                str(self.model_dir),
                device_map=self.device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        except Exception as exc:
            raise DubError(
                f"Could not load Qwen3-TTS weights from {self.model_dir}. "
                f"The checkpoint files are missing or unreadable. {exc}"
            ) from exc
        return self._model


def build_synthesizer(name: str, model_dir: Path | None, openvoice_dir: Path | None, device: str):
    if name == "reference":
        return ReferenceSynthesizer("auto")
    if name == "indextts":
        if model_dir is None:
            raise DubError("IndexTTS needs --tts-model-dir or INDEXTTS_MODEL_DIR.")
        return IndexTTSSynthesizer(model_dir, device)
    if name == "qwen3":
        if model_dir is None:
            raise DubError("Qwen3-TTS needs --tts-model-dir or INDEXTTS_MODEL_DIR.")
        return QwenSynthesizer(model_dir, device)
    if name == "openvoice":
        if openvoice_dir is None:
            raise DubError("OpenVoice needs --openvoice-dir or OPENVOICE_DIR.")
        return OpenVoiceSynthesizer(openvoice_dir, device)
    if name == "auto":
        if model_dir is not None and (model_dir / "config.yaml").is_file():
            return IndexTTSSynthesizer(model_dir, device)
        if openvoice_dir is not None and (openvoice_dir / "converter" / "config.json").is_file():
            return OpenVoiceSynthesizer(openvoice_dir, device)
        raise DubError(
            "No speech model is configured. IndexTTS-2.5 is the voice for a real dub. "
            "Set --tts-model-dir or INDEXTTS_MODEL_DIR to a directory that contains config.yaml. "
            "For clean studio voices, pass --tts qwen3 and a Qwen3-TTS CustomVoice directory. "
            "Missing weights stop here. "
            "--tts reference is an espeak-ng stub for wiring checks and is not a finished dub."
        )
    raise DubError(f"Unknown TTS backend {name!r}. Use auto, indextts, qwen3, openvoice, or reference.")


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None


def harmonic_stack(f0: float, count: int, sample_rate: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    f0 = float(np.clip(f0, 70.0, 500.0))
    time = np.arange(count, dtype=np.float32) / sample_rate
    wave = np.zeros(count, dtype=np.float32)
    for partial, amplitude in enumerate((1.0, 0.55, 0.28, 0.14, 0.07), start=1):
        wave += amplitude * np.sin(2.0 * np.pi * f0 * partial * time)
    return normalize_peak(wave, 0.6)


def _espeak(text: str, target_f0: float, target_duration: float, destination: Path) -> None:
    if not espeak_available():
        raise DubError("espeak-ng is not on PATH.")
    words = max(1, len(text.split()))
    wpm = int(np.clip(round(words / target_duration * 60.0), 80, 260))
    pitch = int(np.clip(round((target_f0 - 70.0) * (99.0 / 280.0)), 0, 99))
    script = destination.with_suffix(".txt")
    script.write_text(text, encoding="utf-8")
    try:
        command = [
            "espeak-ng",
            "-v",
            "en-us",
            "-z",
            "-s",
            str(wpm),
            "-p",
            str(pitch),
            "-w",
            str(destination),
            "-f",
            str(script),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            command = [arg for arg in command if arg not in {"-v", "en-us"}]
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.is_file():
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            raise DubError(f"espeak-ng failed to synthesize a line.\n{detail}")
    finally:
        script.unlink(missing_ok=True)


def _wav_seconds(path: Path) -> float:
    audio = load_mono(path, DIALOGUE_RATE)
    return len(audio) / DIALOGUE_RATE


def _prefer_english_key(keys: list[str]) -> str:
    for key in keys:
        if "newest" in key.lower():
            return key
    for key in keys:
        lowered = key.lower()
        if "en" in lowered and "us" in lowered:
            return key
    for key in keys:
        if key.lower().startswith("en"):
            return key
    return keys[0]


def _source_embedding(checkpoint_dir: Path, speaker_key: str) -> Path:
    stem = speaker_key.lower().replace("_", "-")
    search_roots = [
        checkpoint_dir / "base_speakers" / "ses",
        checkpoint_dir / "ses",
    ]
    for root in search_roots:
        candidate = root / f"{stem}.pth"
        if candidate.is_file():
            return candidate
        if root.is_dir():
            matches = sorted(root.glob("*.pth"))
            for match in matches:
                if "en" in match.stem.lower():
                    return match
            if matches:
                return matches[0]
    raise DubError(
        "Could not find a MeloTTS source embedding (*.pth) under "
        f"{checkpoint_dir}/base_speakers/ses."
    )
