from __future__ import annotations

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
    load_mono,
    normalize_peak,
    pitch_shift,
    write_wav,
)
from anime_dub_tools.models import DubError


def speech_seconds(text: str) -> float:
    words = max(1, len(text.split()))
    return max(0.35, words / 2.6)


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
                "backend=reference/harmonic. espeak-ng is not available, so each line is a "
                "tone at the speaker's pitch with the line's loudness contour. It is not "
                "English speech. Install espeak-ng, or use --tts indextts / openvoice."
            )
        return (
            "backend=reference/espeak-ng. Pronunciation comes from espeak-ng. Pitch and "
            "loudness are taken from this episode's isolated dialogue, not from a voice "
            "library. This is intelligible and robotic. --tts indextts clones a speaking "
            "voice from those same lines and uses each line as the emotion reference."
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


class IndexTTSSynthesizer:
    """IndexTTS-2.5 (or 2.x) cloning. Emotion comes from the line's own audio.

    ``use_qwen_emo`` and ``use_emo_text`` stay off. Duration is requested with
    ``duration_factor`` on 2.5, then ffmpeg fits whatever residual remains.
    """

    name = "indextts"

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self._tts = None
        self._generation = ""
        self.notice = (
            "backend=indextts. Each speaker is cloned from their own isolated lines. "
            "The matching line is the emotion reference, so no emotion classifier is used."
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
        tts, generation = self._load()
        target_duration = max(0.12, float(target_duration))
        factor = float(np.clip(target_duration / speech_seconds(text), 0.5, 2.0))
        raw = destination.with_suffix(".tts.wav")
        try:
            self._infer(tts, generation, text, speaker_reference, emotion_reference, raw, factor)
            produced = _wav_seconds(raw)
            if generation == "2.5" and produced > 0 and not 0.85 <= produced / target_duration <= 1.15:
                adjusted = float(np.clip(factor * (target_duration / produced), 0.5, 2.0))
                self._infer(tts, generation, text, speaker_reference, emotion_reference, raw, adjusted)
            fit_duration(raw, destination, target_duration, DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)

    def _load(self):
        if self._tts is not None:
            return self._tts, self._generation
        config = self.model_dir / "config.yaml"
        if not config.is_file():
            raise DubError(
                f"IndexTTS config.yaml was not found in {self.model_dir}. "
                "Download IndexTTS-2.5 checkpoints into that directory and pass --tts-model-dir."
            )
        generation = "2.5"
        try:
            from indextts.infer_v2_5 import IndexTTS2
        except ImportError:
            try:
                from indextts.infer_v2 import IndexTTS2
                generation = "2"
            except ImportError as exc:
                raise DubError(
                    "IndexTTS is not installed. From a GPU machine:\n"
                    "  pip install git+https://github.com/index-tts/index-tts\n"
                    "Put config.yaml and the 2.5 weights in --tts-model-dir. "
                    "Weights are not downloaded by this pipeline."
                ) from exc
        kwargs = {
            "cfg_path": str(config),
            "model_dir": str(self.model_dir),
            "device": self.device,
        }
        if generation == "2.5":
            kwargs["use_bf16"] = self.device.startswith("cuda")
            kwargs["use_qwen_emo"] = False
        else:
            kwargs["use_fp16"] = False
            kwargs["use_cuda_kernel"] = False
        self._tts = IndexTTS2(**kwargs)
        self._generation = generation
        return self._tts, self._generation

    @staticmethod
    def _infer(tts, generation: str, text: str, speaker: Path, emotion: Path | None, destination: Path, factor: float) -> None:
        kwargs = {
            "spk_audio_prompt": str(speaker),
            "text": text,
            "output_path": str(destination),
            "emo_audio_prompt": str(emotion) if emotion else None,
            "emo_alpha": 0.85,
            "use_emo_text": False,
            "verbose": False,
            "interval_silence": 80,
        }
        if generation == "2.5":
            kwargs["lang"] = "EN"
            kwargs["duration_factor"] = factor
        tts.infer(**kwargs)
        if not destination.is_file():
            raise DubError(f"IndexTTS did not write {destination}")


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
            fit_duration(colored, destination, target_duration, DIALOGUE_RATE)
        finally:
            raw.unlink(missing_ok=True)
            colored.unlink(missing_ok=True)

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


def build_synthesizer(name: str, model_dir: Path | None, openvoice_dir: Path | None, device: str):
    if name == "reference":
        return ReferenceSynthesizer("auto")
    if name == "indextts":
        if model_dir is None:
            raise DubError("IndexTTS needs --tts-model-dir or INDEXTTS_MODEL_DIR.")
        return IndexTTSSynthesizer(model_dir, device)
    if name == "openvoice":
        if openvoice_dir is None:
            raise DubError("OpenVoice needs --openvoice-dir or OPENVOICE_DIR.")
        return OpenVoiceSynthesizer(openvoice_dir, device)
    if name == "auto":
        if model_dir is not None and (model_dir / "config.yaml").is_file():
            return IndexTTSSynthesizer(model_dir, device)
        if openvoice_dir is not None and (openvoice_dir / "converter" / "config.json").is_file():
            return OpenVoiceSynthesizer(openvoice_dir, device)
        return ReferenceSynthesizer("auto")
    raise DubError(f"Unknown TTS backend {name!r}. Use auto, indextts, openvoice, or reference.")


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
