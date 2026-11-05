from __future__ import annotations

from pathlib import Path

import numpy as np

from anime_dub_tools.audio import load_mono
from anime_dub_tools.models import DubError, Turn

DEFAULT_DIARIZER_MODEL = "pyannote/speaker-diarization-community-1"
_SPECTRAL_NOTICE = (
    "backend=spectral. Speakers are clustered from this episode's spectrum. "
    "That keeps identities stable for the file, but it is not a neural diarizer. "
    "For a real episode, install pyannote.audio, accept the user conditions for "
    f"{DEFAULT_DIARIZER_MODEL}, and set HUGGINGFACE_API_KEY (or HF_TOKEN)."
)


class SpectralDiarizer:
    name = "spectral"
    notice = _SPECTRAL_NOTICE

    def diarize(self, audio: Path) -> list[Turn]:
        samples = load_mono(audio, 16000)
        return diarize_samples(samples, 16000)


class PyannoteDiarizer:
    name = "pyannote"
    notice = None

    def __init__(self, model: str, token: str, device: str) -> None:
        self.model = model
        self.token = token
        self.device = device
        self._pipeline = None

    def diarize(self, audio: Path) -> list[Turn]:
        pipeline = self._load()
        output = pipeline(str(audio))
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None or _annotation_empty(annotation):
            annotation = output.speaker_diarization
        turns = []
        for segment, speaker in _iter_annotation(annotation):
            start = float(segment.start)
            end = float(segment.end)
            if end - start <= 0.05:
                continue
            label = str(speaker)
            if label.isdigit():
                label = f"SPEAKER_{int(label):02d}"
            turns.append(Turn(start=start, end=end, speaker=label))
        turns.sort(key=lambda turn: (turn.start, turn.speaker))
        return turns

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DubError(
                "pyannote diarization needs `pip install pyannote.audio` and ffmpeg. "
                f"Accept the conditions for {self.model} and pass a Hugging Face token."
            ) from exc
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=self.token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.token)
        if pipeline is None:
            raise DubError(f"Could not load {self.model}. Check the token and model access.")
        want_cuda = self.device.startswith("cuda")
        if want_cuda and not torch.cuda.is_available():
            raise DubError("CUDA was requested for diarization, but torch.cuda.is_available() is false.")
        if want_cuda or (self.device == "auto" and torch.cuda.is_available()):
            pipeline.to(torch.device("cuda"))
        self._pipeline = pipeline
        return pipeline


def build_diarizer(name: str, token: str | None, model: str | None, device: str):
    chosen_model = model or DEFAULT_DIARIZER_MODEL
    if name == "spectral":
        return SpectralDiarizer()
    if name == "pyannote":
        if not token:
            raise DubError(
                "pyannote diarization needs a Hugging Face token in HUGGINGFACE_API_KEY, "
                "HF_TOKEN, or --hf-token. Accept the model conditions first. "
                "The rest of the pipeline does not require the token: use --diarizer spectral "
                "or leave --diarizer auto."
            )
        return PyannoteDiarizer(chosen_model, token, device)
    if name == "auto":
        if token and _pyannote_available():
            return PyannoteDiarizer(chosen_model, token, device)
        diarizer = SpectralDiarizer()
        if token and not _pyannote_available():
            diarizer.notice = (
                "backend=spectral. A Hugging Face token is set, but pyannote.audio "
                "is not installed, so diarization stayed local. pip install pyannote.audio"
            )
        return diarizer
    raise DubError(f"Unknown diarizer {name!r}. Use auto, pyannote, or spectral.")


def diarize_samples(samples: np.ndarray, sample_rate: int) -> list[Turn]:
    samples = np.asarray(samples, dtype=np.float32)
    duration = len(samples) / sample_rate if sample_rate else 0.0
    if duration < 0.2:
        if duration <= 0:
            return []
        return [Turn(0.0, duration, "SPEAKER_00")]
    window = int(0.030 * sample_rate)
    hop = int(0.010 * sample_rate)
    frame_count = 1 + max(0, (len(samples) - window) // hop)
    rms = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        frame = samples[index * hop : index * hop + window]
        rms[index] = np.sqrt(np.mean(frame * frame) + 1e-12)
    peak = float(rms.max())
    if peak < 1e-4:
        return []
    threshold = max(float(np.percentile(rms, 55)) * 0.45, peak * 0.18)
    voiced = rms >= threshold
    segments: list[tuple[float, float]] = []
    index = 0
    while index < len(voiced):
        if not voiced[index]:
            index += 1
            continue
        end_index = index
        while end_index < len(voiced) and voiced[end_index]:
            end_index += 1
        start = index * hop / sample_rate
        end = min(duration, (end_index * hop + window) / sample_rate)
        if end - start >= 0.20:
            segments.append((start, end))
        index = end_index
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for start, end in segments:
        feature = _spectrum(samples, sample_rate, start, end)
        if not centroids:
            centroids.append(feature)
            labels.append(0)
            continue
        distances = [1.0 - float(np.dot(feature, centroid)) for centroid in centroids]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 0.40:
            labels.append(nearest)
            centroids[nearest] = _unit(centroids[nearest] * 0.75 + feature * 0.25)
        else:
            labels.append(len(centroids))
            centroids.append(feature)
    turns = [
        Turn(start=start, end=end, speaker=f"SPEAKER_{label:02d}")
        for (start, end), label in zip(segments, labels)
    ]
    return _merge_turns(turns)


def _spectrum(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    left = int(start * sample_rate)
    right = max(left + 8, int(end * sample_rate))
    clip = samples[left:right]
    if len(clip) > sample_rate:
        mid = len(clip) // 2
        half = sample_rate // 2
        clip = clip[mid - half : mid + half]
    windowed = clip * np.hanning(len(clip))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(clip), d=1.0 / sample_rate)
    high = min(4000.0, sample_rate / 2 - 1)
    edges = np.geomspace(80.0, high, 25)
    bands = []
    for low, upper in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < upper)
        bands.append(float(np.log1p(spectrum[mask].mean())) if np.any(mask) else 0.0)
    return _unit(np.asarray(bands, dtype=np.float32))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def _merge_turns(turns: list[Turn]) -> list[Turn]:
    if not turns:
        return []
    merged = [turns[0]]
    for turn in turns[1:]:
        previous = merged[-1]
        if turn.speaker == previous.speaker and turn.start - previous.end <= 0.15:
            merged[-1] = Turn(previous.start, turn.end, previous.speaker)
        else:
            merged.append(turn)
    return merged


def _pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


def _annotation_empty(annotation) -> bool:
    if hasattr(annotation, "itertracks"):
        return next(annotation.itertracks(), None) is None
    return len(tuple(annotation)) == 0


def _iter_annotation(annotation):
    if hasattr(annotation, "itertracks"):
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            yield segment, speaker
        return
    for item in annotation:
        if len(item) == 2:
            yield item[0], item[1]
        elif len(item) >= 3:
            yield item[0], item[2]
