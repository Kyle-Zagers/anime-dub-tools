from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from anime_dub_tools.diarize import DEFAULT_DIARIZER_MODEL, build_diarizer
from anime_dub_tools.models import DubError
from anime_dub_tools.pipeline import dub_episode
from anime_dub_tools.separate import DEFAULT_SEPARATOR_MODEL, build_separator
from anime_dub_tools.synthesize import build_synthesizer


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="anime_dub_tools",
        description=(
            "Dub one anime episode into English. The episode audio is the only "
            "voice source. English subtitles supply the words. Speaker names come "
            "from the subtitle name field when it has one, otherwise from diarization."
        ),
    )
    parser.add_argument("--episode", required=True, type=Path, help="Video or audio for one episode.")
    parser.add_argument("--subs", required=True, type=Path, help="English .ass or .srt subtitles.")
    parser.add_argument("--out", required=True, type=Path, help="Dubbed output, usually an .mkv.")
    parser.add_argument("--separator", choices=("auto", "roformer", "copy"), default="auto")
    parser.add_argument("--separator-model", default=DEFAULT_SEPARATOR_MODEL)
    parser.add_argument("--diarizer", choices=("auto", "pyannote", "spectral"), default="auto")
    parser.add_argument("--diarizer-model", default=DEFAULT_DIARIZER_MODEL)
    parser.add_argument("--tts", choices=("auto", "indextts", "openvoice", "reference"), default="auto")
    parser.add_argument("--tts-model-dir", type=Path, default=_env_path("INDEXTTS_MODEL_DIR"))
    parser.add_argument("--openvoice-dir", type=Path, default=_env_path("OPENVOICE_DIR"))
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Prefer the environment.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--keep-work", type=Path, default=None, help="Keep intermediate wavs in this directory.")
    parser.add_argument(
        "--lips",
        action="store_true",
        help="Patch mouths with MuseTalk when one face passes the gate. Off by default.",
    )
    parser.add_argument("--lips-model-dir", type=Path, default=_env_path("MUSETALK_MODEL_DIR"))
    args = parser.parse_args(argv)
    try:
        device = _device(args.device, neural=_needs_torch(args))
        separator = build_separator(args.separator, args.separator_model, device)
        diarizer = build_diarizer(
            args.diarizer,
            _token(args.hf_token),
            args.diarizer_model,
            device,
        )
        synthesizer = build_synthesizer(
            args.tts,
            args.tts_model_dir,
            args.openvoice_dir,
            device,
        )
        dub_kwargs = {
            "separator": separator,
            "diarizer": diarizer,
            "synthesizer": synthesizer,
            "lips": args.lips,
            "lips_model_dir": args.lips_model_dir,
        }
        if args.keep_work is None:
            with tempfile.TemporaryDirectory(prefix="anime-dub-") as temporary:
                dub_episode(
                    args.episode,
                    args.subs,
                    args.out,
                    work_dir=Path(temporary),
                    **dub_kwargs,
                )
        else:
            dub_episode(
                args.episode,
                args.subs,
                args.out,
                work_dir=args.keep_work,
                **dub_kwargs,
            )
    except DubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _token(explicit: str | None) -> str | None:
    return (
        explicit
        or os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def _needs_torch(args) -> bool:
    if args.diarizer == "pyannote":
        return True
    if args.tts in {"indextts", "openvoice"}:
        return True
    if args.device == "cuda":
        return True
    if args.tts == "auto" and args.tts_model_dir and (args.tts_model_dir / "config.yaml").is_file():
        return True
    if args.tts == "auto" and args.openvoice_dir and (args.openvoice_dir / "converter" / "config.json").is_file():
        return True
    if args.diarizer == "auto" and _token(args.hf_token):
        return True
    return False


def _device(preference: str, neural: bool) -> str:
    if preference == "cpu" or not neural:
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if preference == "cuda":
            raise DubError("CUDA was requested but PyTorch is not installed.") from exc
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise DubError("CUDA was requested but no GPU is visible to PyTorch.")
    if torch.cuda.is_available() and preference in {"auto", "cuda"}:
        index = _largest_cuda_index()
        torch.cuda.set_device(index)
        return f"cuda:{index}"
    return "cpu"


def _largest_cuda_index() -> int:
    """Prefer the GPU with the most memory.

    A machine with an integrated GPU and a discrete card enumerates both.
    The smaller device is often index 0.
    """
    import torch

    best = 0
    best_mem = -1
    for index in range(torch.cuda.device_count()):
        memory = int(torch.cuda.get_device_properties(index).total_memory)
        if memory > best_mem:
            best = index
            best_mem = memory
    return best


if __name__ == "__main__":
    raise SystemExit(main())
