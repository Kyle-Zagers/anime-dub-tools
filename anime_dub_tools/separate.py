from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from anime_dub_tools.audio import (
    BED_RATE,
    load_audio,
    pad_trim_2d,
    resample_linear,
    write_audio,
)
from anime_dub_tools.models import DubError

# Kimberley Jensen Mel-Band RoFormer, unwa FT2 bleedless checkpoint.
# The fullness-tuned forks (FT2 without "bleedless", and the Gabox fullness
# instrumentals) are not the default. The vocal estimate is subtracted from
# the whole episode in the mix, and it is not an instrumental stem.
DEFAULT_SEPARATOR_MODEL = "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"


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

    def __init__(self, model_filename: str, device: str = "cpu") -> None:
        self.model_filename = model_filename
        self.device = device
        self._separator = None
        self._output_dir: Path | None = None
        self.notice = (
            "backend=roformer. Kimberley Jensen Mel-Band RoFormer bleedless "
            f"({model_filename}) at {BED_RATE} Hz. The vocal estimate is removed "
            "from the whole episode, and the bed written out is stereo. "
            "Sounds that overlap speech can still be lost with the vocal estimate."
        )

    def separate(self, audio: Path, work_dir: Path) -> tuple[Path, Path]:
        work_dir.mkdir(parents=True, exist_ok=True)
        mix, sample_rate = load_audio(audio)
        mix = _at_bed_rate(mix, sample_rate)
        pair, _extra = _stereo_pair(mix)
        pair_path = work_dir / "stereo_pair.wav"
        write_audio(pair_path, pair, BED_RATE)
        separator = self._load(work_dir)
        try:
            produced = separator.separate(str(pair_path))
        except DubError:
            raise
        except Exception as exc:
            raise DubError(
                f"Vocal separation failed for {self.model_filename}. "
                "The stereo bed was not replaced with an instrumental stem. "
                f"{exc}"
            ) from exc
        paths = [self._resolve(item, work_dir) for item in produced]
        vocal_path = _first(paths, "vocals")
        if vocal_path is None:
            raise DubError(
                "The separator did not return a vocal estimate, so the bed was not built. "
                "An instrumental or music/effects stem is not used in its place. "
                f"Files: {', '.join(path.name for path in paths) or 'none'}."
            )
        vocals, vocal_rate = load_audio(vocal_path)
        vocals = _at_bed_rate(vocals, vocal_rate)
        kept_vocals = work_dir / "vocals.wav"
        write_audio(kept_vocals, pad_trim_2d(_as_stereo(vocals), pair.shape[0]), BED_RATE)
        return kept_vocals, audio

    def _load(self, work_dir: Path):
        if self._separator is not None and self._output_dir == work_dir:
            return self._separator
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise DubError(_separator_install_message()) from exc
        use_directml = self.device.startswith("privateuseone") or self.device == "directml"
        separator = Separator(
            log_level=logging.WARNING,
            output_dir=str(work_dir),
            output_format="WAV",
            sample_rate=BED_RATE,
            output_single_stem="Vocals",
            use_soundfile=True,
            use_directml=use_directml,
        )
        if self.device.startswith("cuda") or self.device.startswith("privateuseone"):
            import torch

            separator.torch_device = torch.device(self.device)
        try:
            separator.load_model(model_filename=self.model_filename)
        except DubError:
            raise
        except Exception as exc:
            raise DubError(
                f"Could not load vocal separation weights {self.model_filename}. "
                "Install audio-separator and download the Kimberley Jensen Mel-Band "
                "RoFormer bleedless checkpoint, or pass --separator-model. "
                f"Underlying error: {exc}"
            ) from exc
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


def build_separator(name: str, model: str | None, device: str = "cpu"):
    chosen = name
    if name == "auto":
        chosen = "roformer" if separator_available() else "missing"
    if chosen == "copy":
        return CopySeparator()
    if chosen == "roformer":
        if not separator_available():
            raise DubError(_separator_install_message())
        return RoformerSeparator(model or DEFAULT_SEPARATOR_MODEL, device)
    if chosen == "missing":
        raise DubError(_separator_install_message())
    raise DubError(f"Unknown separator {name!r}. Use auto, roformer, or copy.")


def _separator_install_message() -> str:
    return (
        "Vocal isolation needs audio-separator "
        '(pip install "audio-separator[cpu]", or "audio-separator[gpu]" on NVIDIA). '
        f"The default checkpoint is {DEFAULT_SEPARATOR_MODEL}, the Kimberley Jensen "
        "Mel-Band RoFormer bleedless model, at 44.1 kHz stereo. It downloads on first "
        "use and is not gated. Pass --separator copy only when you intentionally want "
        "the original mix left under the dub."
    )


def _at_bed_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == BED_RATE:
        return np.asarray(audio, dtype=np.float32)
    count = max(1, int(round(audio.shape[0] * BED_RATE / sample_rate)))
    columns = [resample_linear(audio[:, channel], count) for channel in range(audio.shape[1])]
    return np.stack(columns, axis=1)


def _stereo_pair(mix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if mix.shape[1] == 1:
        return np.repeat(mix, 2, axis=1), None
    extra = mix[:, 2:] if mix.shape[1] > 2 else None
    return mix[:, :2], extra


def _as_stereo(vocals: np.ndarray) -> np.ndarray:
    if vocals.shape[1] == 1:
        return np.repeat(vocals, 2, axis=1)
    return vocals[:, :2]


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
