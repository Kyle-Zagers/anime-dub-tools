from __future__ import annotations

import logging
from pathlib import Path

from anime_dub_tools.audio import load_mono, media_duration, slice_samples, write_wav
from anime_dub_tools.models import DubError

# Mel-Band RoFormer is the current audio-separator Python default and is aimed
# at vocal quality. Swap it with --separator-model, including other UVR names.
DEFAULT_SEPARATOR_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"


class CopySeparator:
    name = "copy"
    notice = (
        "backend=copy. Vocal isolation is off, so the original mix, including "
        "the source dialogue, stays under the new lines. Use --separator roformer "
        "or install audio-separator for a clean music and effects bed."
    )

    def separate(self, audio: Path, work_dir: Path) -> tuple[Path, Path]:
        del work_dir
        return audio, audio


class RoformerSeparator:
    name = "roformer"
    notice = None

    def __init__(self, model_filename: str) -> None:
        self.model_filename = model_filename
        self._separator = None
        self._output_dir: Path | None = None

    def separate(self, audio: Path, work_dir: Path) -> tuple[Path, Path]:
        work_dir.mkdir(parents=True, exist_ok=True)
        separator = self._load(work_dir)
        produced = separator.separate(str(audio))
        paths = [self._resolve(item, work_dir) for item in produced]
        vocals = _first(paths, "vocals")
        background = _first(paths, "background")
        if vocals is None and background is None:
            raise DubError(f"Separator did not return vocal or instrumental stems: {paths}")
        if vocals is None and background is not None:
            vocals = work_dir / "vocals.wav"
            _subtract(audio, background, vocals)
        elif background is None and vocals is not None:
            background = work_dir / "background.wav"
            _subtract(audio, vocals, background)
        assert vocals is not None and background is not None
        return vocals, background

    def _load(self, work_dir: Path):
        if self._separator is not None and self._output_dir == work_dir:
            return self._separator
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise DubError(_separator_install_message()) from exc
        separator = Separator(
            log_level=logging.WARNING,
            output_dir=str(work_dir),
            output_format="WAV",
            use_soundfile=True,
        )
        separator.load_model(model_filename=self.model_filename)
        self._separator = separator
        self._output_dir = work_dir
        return separator

    @staticmethod
    def _resolve(raw: str, work_dir: Path) -> Path:
        path = Path(raw)
        if path.is_file():
            return path
        candidate = work_dir / raw
        if candidate.is_file():
            return candidate
        matches = list(work_dir.glob(path.name))
        if matches:
            return matches[0]
        raise DubError(f"Separator reported missing output {raw}")


def separator_available() -> bool:
    try:
        import audio_separator.separator  # noqa: F401
    except ImportError:
        return False
    return True


def build_separator(name: str, model: str | None):
    chosen = name
    if name == "auto":
        chosen = "roformer" if separator_available() else "missing"
    if chosen == "copy":
        return CopySeparator()
    if chosen == "roformer":
        if not separator_available():
            raise DubError(_separator_install_message())
        return RoformerSeparator(model or DEFAULT_SEPARATOR_MODEL)
    if chosen == "missing":
        raise DubError(_separator_install_message())
    raise DubError(f"Unknown separator {name!r}. Use auto, roformer, or copy.")


def _separator_install_message() -> str:
    return (
        "Vocal isolation needs audio-separator "
        '(pip install "audio-separator[cpu]", or "audio-separator[gpu]" on NVIDIA). '
        "The Mel-Band RoFormer weights download on first use into the audio-separator "
        "cache. They are not gated. Pass --separator copy only when you intentionally "
        "want the original mix left under the dub."
    )


def _kind(path: Path) -> str:
    name = path.stem.lower()
    if any(token in name for token in ("instrument", "no_vocal", "karaoke", "bgm")):
        return "background"
    if "vocal" in name:
        return "vocals"
    return ""


def _first(paths: list[Path], kind: str) -> Path | None:
    for path in paths:
        if _kind(path) == kind:
            return path
    return None


def _subtract(mix: Path, stem: Path, destination: Path) -> None:
    sample_rate = 44100
    mix_audio = load_mono(mix, sample_rate)
    stem_audio = load_mono(stem, sample_rate)
    count = min(len(mix_audio), len(stem_audio))
    write_wav(destination, mix_audio[:count] - stem_audio[:count], sample_rate)
    if media_duration(destination) <= 0:
        raise DubError(f"Could not build {destination.name} from the separated stem.")
