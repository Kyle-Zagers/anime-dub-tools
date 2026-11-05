"""Optional mouth patches. Off unless --lips is set and MuseTalk weights exist.

A frame is patched only when one face is large, near-frontal, has a visible
uncovered mouth, and that face is the speaker. Every other frame is copied.
The portrait outside the mouth box is never rewritten.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anime_dub_tools.audio import run_cmd
from anime_dub_tools.models import DubError

MIN_FACE_HEIGHT = 0.18
MAX_YAW_DEGREES = 20.0
MOUTH_CLASSES = {11, 12, 13}


@dataclass(frozen=True)
class FaceGate:
    """One face, with the measurements the gate needs.

    ``bbox`` is ``x, y, width, height`` as fractions of the frame.
    ``yaw_degrees`` is 0 when the face is frontal. ``is_speaker`` is true only
    when this face is the person who has a line at this frame.
    """

    bbox: tuple[float, float, float, float]
    yaw_degrees: float
    mouth_visible: bool
    mouth_covered: bool
    is_speaker: bool


def select_patch(faces: list[FaceGate], line_active: bool) -> FaceGate | None:
    """Return the one face whose mouth may be patched, or nothing."""
    if not line_active or len(faces) != 1:
        return None
    face = faces[0]
    if not face.is_speaker:
        return None
    if face.bbox[3] < MIN_FACE_HEIGHT:
        return None
    if abs(face.yaw_degrees) > MAX_YAW_DEGREES:
        return None
    if not face.mouth_visible or face.mouth_covered:
        return None
    return face


def mouth_box_pixels(face: FaceGate, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Lower-face mouth rectangle. The eyes and the rest of the head stay outside it."""
    x, y, width, height = face.bbox
    left = int(round((x + 0.2 * width) * frame_w))
    top = int(round((y + 0.62 * height) * frame_h))
    box_w = max(1, int(round(0.6 * width * frame_w)))
    box_h = max(1, int(round(0.28 * height * frame_h)))
    left = min(max(0, left), max(0, frame_w - 1))
    top = min(max(0, top), max(0, frame_h - 1))
    box_w = min(box_w, frame_w - left)
    box_h = min(box_h, frame_h - top)
    return left, top, max(1, box_w), max(1, box_h)


def composite_mouth_only(original: np.ndarray, edited: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Copy ``edited``'s mouth box onto ``original`` and leave every other pixel."""
    output = np.array(original, copy=True)
    x, y, width, height = box
    output[y : y + height, x : x + width] = edited[y : y + height, x : x + width]
    return output


def line_active_at(time_s: float, spans: list[tuple[float, float]]) -> bool:
    return any(start <= time_s < end for start, end in spans)


def render_gated_frames(
    frames: list[np.ndarray],
    times: list[float],
    faces_at: list[list[FaceGate]],
    spans: list[tuple[float, float]],
    patch,
) -> list[np.ndarray]:
    """Apply ``patch(frame, mouth_box)`` only on frames that pass the gate."""
    rendered: list[np.ndarray] = []
    for frame, time_s, faces in zip(frames, times, faces_at):
        chosen = select_patch(faces, line_active_at(time_s, spans))
        if chosen is None:
            rendered.append(np.array(frame, copy=True))
            continue
        box = mouth_box_pixels(chosen, frame.shape[1], frame.shape[0])
        edited = patch(frame, box)
        rendered.append(composite_mouth_only(frame, edited, box))
    return rendered


def lips_skip_message(model_dir: Path | None) -> str | None:
    """None when mouth patching can run. Otherwise a reason to copy the picture."""
    missing = missing_weight_notes(model_dir)
    if missing:
        detail = " ".join(missing)
        return (
            "MuseTalk weights are missing, so --lips copied the picture unchanged. "
            f"{detail} Mouths are patched only when those weights are present."
        )
    try:
        import musetalk.utils.utils  # noqa: F401
    except ImportError:
        return (
            "MuseTalk is not installed, so --lips copied the picture unchanged. "
            "Install MuseTalk next to its weights. The portrait was not regenerated."
        )
    return None


def missing_weight_notes(model_dir: Path | None) -> list[str]:
    if model_dir is None:
        return [
            "Set --lips-model-dir or MUSETALK_MODEL_DIR to the MuseTalk weights directory."
        ]
    if not model_dir.is_dir():
        return [f"{model_dir} is not a directory."]
    notes = []
    if _first_file(model_dir, ("musetalkV15/unet.pth", "musetalk/pytorch_model.bin", "unet.pth")) is None:
        notes.append("Missing unet weights (musetalkV15/unet.pth or musetalk/pytorch_model.bin).")
    if _first_file(model_dir, ("musetalkV15/musetalk.json", "musetalk/config.json", "config.json")) is None:
        notes.append("Missing the unet config (musetalk.json or config.json).")
    whisper = _whisper_dir(model_dir)
    if whisper is None:
        notes.append("Missing the whisper directory (models/whisper).")
    return notes


def render_lips(
    episode: Path,
    audio: Path,
    destination: Path,
    model_dir: Path,
    spans: list[tuple[float, float]],
) -> tuple[int, int]:
    """Write a picture track. Raises DubError if weights were present but inference failed."""
    session = _MuseTalkSession(model_dir, audio)
    try:
        return _stream_frames(episode, destination, spans, session)
    except DubError:
        raise
    except Exception as exc:
        raise DubError(
            "MuseTalk weights were found, but lip sync failed before a picture was finished. "
            f"{exc}"
        ) from exc


def _stream_frames(episode, destination, spans, session) -> tuple[int, int]:
    width, height, fps = _probe_video(episode)
    frame_bytes = width * height * 3
    destination.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(episode),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    patched = 0
    copied = 0
    index = 0
    assert decoder.stdout is not None and encoder.stdin is not None
    try:
        while True:
            raw = _read_exact(decoder.stdout, frame_bytes)
            if raw is None:
                break
            if len(raw) != frame_bytes:
                raise DubError("The picture stream ended in the middle of a frame.")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            time_s = index / fps
            active = line_active_at(time_s, spans)
            faces = session.observe(frame, active)
            chosen = select_patch(faces, active)
            if chosen is None:
                output = frame
                copied += 1
            else:
                box = mouth_box_pixels(chosen, width, height)
                edited = session.paint_mouth(frame, box, time_s, fps)
                output = composite_mouth_only(frame, edited, box)
                patched += 1
            encoder.stdin.write(np.ascontiguousarray(output).tobytes())
            index += 1
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
    if decoder.wait() != 0 or encoder.wait() != 0:
        raise DubError("ffmpeg could not read or write the picture while applying mouth patches.")
    if index == 0:
        raise DubError("The episode picture had no frames to copy.")
    print(f"[lips] patched_frames={patched} copied_frames={copied}", flush=True)
    return patched, copied


class _MuseTalkSession:
    """Lazy MuseTalk loader. Mouth pixels are the only ones allowed to change."""

    def __init__(self, model_dir: Path, audio: Path) -> None:
        self.model_dir = model_dir
        self.audio_path = audio
        self._ready = None

    def observe(self, frame: np.ndarray, line_active: bool) -> list[FaceGate]:
        if not line_active:
            return []
        try:
            return self._observe(frame)
        except DubError:
            raise
        except Exception as exc:
            raise DubError(f"MuseTalk could not measure the face in a frame. {exc}") from exc

    def paint_mouth(self, frame: np.ndarray, box: tuple[int, int, int, int], time_s: float, fps: float) -> np.ndarray:
        try:
            return self._paint(frame, box, time_s, fps)
        except DubError:
            raise
        except Exception as exc:
            raise DubError(f"MuseTalk could not patch a mouth. The portrait was not saved. {exc}") from exc

    def _load(self):
        if self._ready is not None:
            return self._ready
        import torch
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.utils import load_all_model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        unet_path = _first_file(
            self.model_dir, ("musetalkV15/unet.pth", "musetalk/pytorch_model.bin", "unet.pth")
        )
        config_path = _first_file(
            self.model_dir, ("musetalkV15/musetalk.json", "musetalk/config.json", "config.json")
        )
        whisper_dir = _whisper_dir(self.model_dir)
        if unet_path is None or config_path is None or whisper_dir is None:
            raise DubError("MuseTalk weights disappeared between the check and the load.")
        vae, unet, pe = load_all_model(
            unet_model_path=str(unet_path),
            vae_type="sd-vae",
            unet_config=str(config_path),
            device=device,
        )
        timesteps = torch.tensor([0], device=device)
        use_half = device.type == "cuda"
        if use_half:
            pe = pe.half()
            vae.vae = vae.vae.half()
            unet.model = unet.model.half()
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
        self._ready = {
            "device": device,
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "timesteps": timesteps,
            "audio": audio_processor,
            "parsing": FaceParsing(),
            "whisper_dir": whisper_dir,
        }
        return self._ready

    def _observe(self, frame: np.ndarray) -> list[FaceGate]:
        import cv2
        from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox

        runtime = self._load()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = Path(handle.name)
        try:
            if not cv2.imwrite(str(image_path), bgr):
                raise DubError("Could not write a frame for face measurement.")
            coords, frames = get_landmark_and_bbox([str(image_path)], 0)
        finally:
            image_path.unlink(missing_ok=True)
        faces: list[FaceGate] = []
        height, width = frame.shape[:2]
        for bbox, crop_source in zip(coords, frames):
            if _is_placeholder(bbox, coord_placeholder):
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = crop_source[y1:y2, x1:x2]
            mask = _parse_mask(runtime["parsing"], crop)
            faces.append(_face_from_mask((x1, y1, x2, y2), (height, width), mask))
        return faces

    def _paint(self, frame: np.ndarray, box: tuple[int, int, int, int], time_s: float, fps: float) -> np.ndarray:
        import cv2
        from musetalk.utils.utils import datagen
        from transformers import WhisperModel

        runtime = self._load()
        if "whisper" not in runtime:
            weight_dtype = runtime["unet"].model.dtype
            whisper = WhisperModel.from_pretrained(str(runtime["whisper_dir"]))
            whisper = whisper.to(device=runtime["device"], dtype=weight_dtype).eval()
            whisper.requires_grad_(False)
            runtime["whisper"] = whisper
        # The official feature call needs a file. Reuse the episode mix path stored by render.
        features, length = runtime["audio"].get_audio_feature(str(self.audio_path))
        chunks = runtime["audio"].get_whisper_chunk(
            features,
            runtime["device"],
            runtime["unet"].model.dtype,
            runtime["whisper"],
            length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        frame_index = min(int(time_s * fps), max(0, len(chunks) - 1))
        x, y, width, height = box
        # Give the model the face-sized context it was trained on, then keep only the mouth.
        pad = int(max(width, height))
        top = max(0, y - pad)
        left = max(0, x - pad)
        bottom = min(frame.shape[0], y + height + pad // 2)
        right = min(frame.shape[1], x + width + pad)
        crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_RGB2BGR)
        if crop.size == 0:
            return frame
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = runtime["vae"].get_latents_for_unet(resized)
        generator = datagen(
            whisper_chunks=[chunks[frame_index]],
            vae_encode_latents=[latents],
            batch_size=1,
            delay_frame=0,
            device=runtime["device"],
        )
        whisper_batch, latent_batch = next(iter(generator))
        audio_feature = runtime["pe"](whisper_batch)
        latent_batch = latent_batch.to(dtype=runtime["unet"].model.dtype)
        predicted = runtime["unet"].model(
            latent_batch, runtime["timesteps"], encoder_hidden_states=audio_feature
        ).sample
        restored = runtime["vae"].decode_latents(predicted)[0]
        restored = cv2.resize(restored.astype(np.uint8), (right - left, bottom - top))
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        full = np.array(frame, copy=True)
        full[top:bottom, left:right] = restored
        return full


def _is_placeholder(bbox, placeholder) -> bool:
    if bbox is None:
        return True
    try:
        return bool(np.array_equal(np.asarray(bbox), np.asarray(placeholder)))
    except (TypeError, ValueError):
        return False


def _face_from_mask(bbox: tuple[int, int, int, int], frame_hw: tuple[int, int], mask: np.ndarray | None) -> FaceGate:
    frame_h, frame_w = frame_hw
    x1, y1, x2, y2 = bbox
    height = max(1, y2 - y1)
    width = max(1, x2 - x1)
    mouth_pixels = 0
    yaw = 90.0
    if mask is not None and mask.size:
        mouth_pixels = int(np.isin(mask, list(MOUTH_CLASSES)).sum())
        face = mask > 0
        mid = face.shape[1] // 2
        left = int(face[:, :mid].sum())
        right = int(face[:, mid:].sum())
        balance = abs(left - right) / (left + right + 1e-6)
        yaw = float(balance * 80.0)
    mouth_visible = mouth_pixels >= 40
    return FaceGate(
        bbox=(x1 / frame_w, y1 / frame_h, width / frame_w, height / frame_h),
        yaw_degrees=yaw,
        mouth_visible=mouth_visible,
        mouth_covered=not mouth_visible,
        is_speaker=True,
    )


def _parse_mask(parser, crop: np.ndarray) -> np.ndarray | None:
    if crop.size == 0:
        return None
    try:
        parsed = parser(crop)
    except TypeError:
        parsed = parser.parse(crop)
    array = np.asarray(parsed)
    if array.ndim == 3:
        array = array.argmax(axis=0) if array.shape[0] < array.shape[-1] else array.argmax(axis=-1)
    return array


def _probe_video(path: Path) -> tuple[int, int, float]:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise DubError(f"Could not read the picture size of {path}")
    width, height = int(lines[0]), int(lines[1])
    rate = lines[2]
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) else 0.0
    else:
        fps = float(rate)
    if width <= 0 or height <= 0 or fps <= 0:
        raise DubError(f"Could not read the picture size of {path}")
    return width, height, fps


def _read_exact(stream, count: int) -> bytes | None:
    chunks = []
    remaining = count
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            return None if not chunks else b"".join(chunks)
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _first_file(root: Path, relatives: tuple[str, ...]) -> Path | None:
    for relative in relatives:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _whisper_dir(root: Path) -> Path | None:
    for relative in ("whisper", "models/whisper"):
        candidate = root / relative
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    return None
