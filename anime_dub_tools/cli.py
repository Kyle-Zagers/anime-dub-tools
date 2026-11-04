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
    args = parser.parse_args(argv)
    try:
        separator = build_separator(args.separator, args.separator_model)
        diarizer = build_diarizer(
            args.diarizer,
            _token(args.hf_token),
            args.diarizer_model,
            _device(args.device, neural=_needs_torch(args)),
        )
        synthesizer = build_synthesizer(
            args.tts,
            args.tts_model_dir,
            args.openvoice_dir,
            _device(args.device, neural=_needs_torch(args)),
        )
        if args.keep_work is None:
            with tempfile.TemporaryDirectory(prefix="anime-dub-") as temporary:
                dub_episode(
                    args.episode,
                    args.subs,
                    args.out,
                    separator=separator,
                    diarizer=diarizer,
                    synthesizer=synthesizer,
                    work_dir=Path(temporary),
                )
        else:
            dub_episode(
                args.episode,
                args.subs,
                args.out,
                separator=separator,
                diarizer=diarizer,
                synthesizer=synthesizer,
                work_dir=args.keep_work,
            )
    except DubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


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
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
