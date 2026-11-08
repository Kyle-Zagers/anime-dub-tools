"""One English preset per Japanese voice, stable for the episode.

Each subtitle line is its own cue. The cues are clustered inside a scene,
the same voice is linked across scenes, and a line keeps the previous voice
unless its Japanese embedding is clearly closer to someone else. A female
preset is used only when that cluster's pitch stays high. A shout on one
line does not change the voice.
"""

from __future__ import annotations

import numpy as np

MALE_PRESETS = ("Ryan", "Aiden", "Uncle_Fu", "Eric", "Dylan")
FEMALE_PRESETS = ("Vivian", "Serena", "Ono_Anna", "Sohee")

# On the isolated vocal, the nearest other line sat near 0.33 cosine
# distance at the median. 0.55 joins a person's own lines. It does not
# join two voices that are actually far apart.
SAME_SPEAKER_DISTANCE = 0.55
STICK_MARGIN = 0.12
SCENE_GAP_SECONDS = 3.0
FEMALE_MIN_LINES = 4
FEMALE_MEDIAN_HZ = 210.0
FEMALE_FLOOR_HZ = 185.0


def stable_voices(
    embeddings: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    pitches: list[float | None],
    *,
    same_speaker: float = SAME_SPEAKER_DISTANCE,
    stick_margin: float = STICK_MARGIN,
    scene_gap: float = SCENE_GAP_SECONDS,
) -> list[str]:
    """Preset name per cue. The same name is the same voice for the episode."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be (cues, dimensions)")
    count = vectors.shape[0]
    if count == 0:
        return []
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    finite = np.isfinite(vectors).all(axis=1)
    local = _scene_clusters(vectors, starts, ends, finite, same_speaker, scene_gap)
    linked = _link_scenes(local, vectors, finite, same_speaker)
    linked = _assign_nearest(linked, vectors, finite)
    linked = _stick(linked, vectors, finite, stick_margin)
    linked = _fill_missing(linked)
    return _cast(linked, vectors, finite, pitches)


def scene_groups(starts: np.ndarray, ends: np.ndarray, gap: float) -> list[list[int]]:
    """Cues separated by more than `gap` seconds of silence are different scenes."""
    groups: list[list[int]] = []
    current: list[int] = []
    previous_end: float | None = None
    for index, (start, end) in enumerate(zip(starts, ends)):
        if current and previous_end is not None and float(start) - previous_end > gap:
            groups.append(current)
            current = []
        current.append(index)
        previous_end = float(end)
    if current:
        groups.append(current)
    return groups


def _scene_clusters(
    vectors: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    finite: np.ndarray,
    threshold: float,
    gap: float,
) -> np.ndarray:
    labels = np.full(len(vectors), -1, dtype=int)
    next_id = 0
    for group in scene_groups(starts, ends, gap):
        usable = [index for index in group if finite[index]]
        if not usable:
            continue
        durations = [float(ends[index] - starts[index]) for index in usable]
        local = _greedy(vectors[usable], durations, threshold)
        for index, label in zip(usable, local):
            labels[index] = next_id + int(label)
        next_id += int(local.max()) + 1
    return labels


def _greedy(vectors: np.ndarray, durations: list[float], threshold: float) -> np.ndarray:
    """Longer cues open a voice. A later cue joins one when it is close."""
    labels = np.full(len(vectors), -1, dtype=int)
    centroids: list[np.ndarray] = []
    order = sorted(range(len(vectors)), key=lambda index: durations[index], reverse=True)
    for index in order:
        vector = _unit(vectors[index])
        if not centroids:
            centroids.append(vector)
            labels[index] = 0
            continue
        distances = 1.0 - np.stack(centroids) @ vector
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= threshold:
            labels[index] = nearest
            members = np.stack([_unit(vectors[row]) for row in range(len(labels)) if labels[row] == nearest])
            centroids[nearest] = _unit(members.mean(axis=0))
        else:
            labels[index] = len(centroids)
            centroids.append(vector)
    return labels


def _link_scenes(
    local: np.ndarray,
    vectors: np.ndarray,
    finite: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """The same Japanese voice in a later scene keeps the earlier id."""
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(local):
        if label < 0 or not finite[index]:
            continue
        clusters.setdefault(int(label), []).append(index)
    if not clusters:
        return np.full(len(local), -1, dtype=int)
    order = sorted(clusters, key=lambda label: len(clusters[label]), reverse=True)
    groups: list[list[int]] = []
    mapping: dict[int, int] = {}
    for label in order:
        members = clusters[label]
        centroid = _centroid(vectors, members)
        if not groups:
            groups.append(list(members))
            mapping[label] = 0
            continue
        distances = [
            _cosine_distance(centroid, _centroid(vectors, group)) for group in groups
        ]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= threshold:
            groups[nearest].extend(members)
            mapping[label] = nearest
        else:
            mapping[label] = len(groups)
            groups.append(list(members))
    linked = np.full(len(local), -1, dtype=int)
    for index, label in enumerate(local):
        if label < 0:
            continue
        linked[index] = mapping[int(label)]
    return linked


def _assign_nearest(labels: np.ndarray, vectors: np.ndarray, finite: np.ndarray) -> np.ndarray:
    centroids, ids = _centroid_table(labels, vectors, finite)
    if not ids:
        return labels
    stacked = np.stack([centroids[speaker] for speaker in ids])
    assigned = labels.copy()
    for index in range(len(labels)):
        if not finite[index]:
            continue
        distances = 1.0 - stacked @ _unit(vectors[index])
        assigned[index] = ids[int(np.argmin(distances))]
    return assigned


def _stick(
    labels: np.ndarray,
    vectors: np.ndarray,
    finite: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Keep the current voice when the next line is only slightly closer to someone else."""
    stuck = labels.copy()
    previous: int | None = None
    for index in range(len(stuck)):
        if not finite[index]:
            if previous is not None:
                stuck[index] = previous
            continue
        centroids, ids = _centroid_table(labels, vectors, finite, skip=index)
        if not ids:
            if previous is not None:
                stuck[index] = previous
            continue
        position = {speaker: slot for slot, speaker in enumerate(ids)}
        stacked = np.stack([centroids[speaker] for speaker in ids])
        distances = 1.0 - stacked @ _unit(vectors[index])
        best = ids[int(np.argmin(distances))]
        if previous is not None and previous in position:
            if float(distances[position[previous]]) <= float(distances[position[best]]) + margin:
                best = previous
        stuck[index] = best
        previous = int(best)
    return stuck


def _fill_missing(labels: np.ndarray) -> np.ndarray:
    filled = labels.copy()
    previous = -1
    for index, label in enumerate(filled):
        if label >= 0:
            previous = int(label)
        elif previous >= 0:
            filled[index] = previous
    following = -1
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] >= 0:
            following = int(filled[index])
        elif following >= 0:
            filled[index] = following
    return filled


def _cast(
    labels: np.ndarray,
    vectors: np.ndarray,
    finite: np.ndarray,
    pitches: list[float | None],
) -> list[str]:
    speakers = [int(speaker) for speaker in dict.fromkeys(labels.tolist()) if int(speaker) >= 0]
    if not speakers:
        return [MALE_PRESETS[0]] * len(labels)
    female = {speaker for speaker in speakers if _confident_female(labels, pitches, speaker)}
    male = [speaker for speaker in speakers if speaker not in female]
    male.sort(key=lambda speaker: int(np.sum(labels == speaker)), reverse=True)
    female_ids = sorted(female, key=lambda speaker: int(np.sum(labels == speaker)), reverse=True)
    names = _preset_names(male, vectors, finite, labels, MALE_PRESETS)
    names.update(_preset_names(female_ids, vectors, finite, labels, FEMALE_PRESETS))
    fallback = names.get(male[0], MALE_PRESETS[0]) if male else names.get(female_ids[0], MALE_PRESETS[0])
    return [names.get(int(label), fallback) for label in labels]


def _confident_female(labels: np.ndarray, pitches: list[float | None], speaker: int) -> bool:
    """True only when the cluster is high for the whole group, not for one shout."""
    values = [
        float(pitch)
        for index, pitch in enumerate(pitches)
        if int(labels[index]) == speaker and pitch is not None
    ]
    if len(values) < FEMALE_MIN_LINES:
        return False
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 25)) >= FEMALE_FLOOR_HZ and float(np.median(array)) >= FEMALE_MEDIAN_HZ


def _preset_names(
    speakers: list[int],
    vectors: np.ndarray,
    finite: np.ndarray,
    labels: np.ndarray,
    presets: tuple[str, ...],
) -> dict[int, str]:
    """The busiest clusters take the distinct presets. An extra cluster shares the nearest one."""
    names: dict[int, str] = {}
    seated: list[int] = []
    for speaker in speakers:
        if len(seated) < len(presets):
            names[speaker] = presets[len(seated)]
            seated.append(speaker)
            continue
        centroid = _speaker_centroid(vectors, finite, labels, speaker)
        distances = [
            _cosine_distance(centroid, _speaker_centroid(vectors, finite, labels, other))
            for other in seated
        ]
        names[speaker] = names[seated[int(np.argmin(distances))]]
    return names


def _centroid_table(
    labels: np.ndarray,
    vectors: np.ndarray,
    finite: np.ndarray,
    skip: int | None = None,
) -> tuple[dict[int, np.ndarray], list[int]]:
    ids = sorted(
        {
            int(label)
            for index, (label, ok) in enumerate(zip(labels, finite))
            if ok and int(label) >= 0 and index != skip
        }
    )
    centroids = {
        speaker: _speaker_centroid(vectors, finite, labels, speaker, skip=skip) for speaker in ids
    }
    return centroids, ids


def _speaker_centroid(
    vectors: np.ndarray,
    finite: np.ndarray,
    labels: np.ndarray,
    speaker: int,
    skip: int | None = None,
) -> np.ndarray:
    members = [
        index
        for index, (label, ok) in enumerate(zip(labels, finite))
        if ok and int(label) == speaker and index != skip
    ]
    if not members:
        members = [index for index, label in enumerate(labels) if int(label) == speaker]
    return _centroid(vectors, members)


def _centroid(vectors: np.ndarray, members: list[int]) -> np.ndarray:
    stacked = np.stack([_unit(vectors[index]) for index in members])
    return _unit(stacked.mean(axis=0))


def embed_cues(samples: np.ndarray, sample_rate: int, spans: list[tuple[float, float]], embedder) -> np.ndarray:
    """One speaker embedding per cue from the isolated Japanese.

    A cue that is silence, or too short to embed, is a row of NaN. The
    clusterer leaves those lines to the neighbor.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    dimension = int(embedder.dimension)
    minimum = int(embedder.min_num_samples)
    rows = []
    for start, end in spans:
        left = int(start * sample_rate)
        right = max(left + 1, int(end * sample_rate))
        clip = audio[left:right]
        energy = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
        if energy < 10 ** (-46 / 20) or len(clip) < minimum // 2:
            rows.append(np.full(dimension, np.nan, dtype=np.float32))
            continue
        clip = _loudest(clip, sample_rate)
        if len(clip) < minimum:
            clip = np.pad(clip, (0, minimum - len(clip)))
        vector = _embed_clip(embedder, clip)
        rows.append(_unit(vector).astype(np.float32))
    if not rows:
        return np.zeros((0, dimension), dtype=np.float32)
    return np.stack(rows)


def load_speaker_embedder(token: str | None, device: str):
    """The embedding packed with the pyannote community-1 diarizer."""
    try:
        import torch
        from pyannote.audio.pipelines.speaker_verification import (
            PyannoteAudioPretrainedSpeakerEmbedding,
        )
    except ImportError as exc:
        from anime_dub_tools.models import DubError

        raise DubError(
            "Voice identity needs pyannote.audio so each Japanese line can be embedded. "
            "pip install pyannote.audio"
        ) from exc
    torch_device = torch.device(device if str(device).startswith("cuda") else "cpu")
    return PyannoteAudioPretrainedSpeakerEmbedding(
        {
            "checkpoint": "pyannote/speaker-diarization-community-1",
            "subfolder": "embedding",
        },
        device=torch_device,
        token=token,
    )


def _embed_clip(embedder, clip: np.ndarray) -> np.ndarray:
    import torch

    wave = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).reshape(1, 1, -1)
    return np.asarray(embedder(wave)[0], dtype=np.float64)


def _loudest(clip: np.ndarray, sample_rate: int, seconds: float = 2.5) -> np.ndarray:
    window = int(seconds * sample_rate)
    if len(clip) <= window:
        return clip
    hop = max(1, int(0.1 * sample_rate))
    best_at = 0
    best = -1.0
    for start in range(0, len(clip) - window + 1, hop):
        energy = float(np.sqrt(np.mean(np.square(clip[start : start + window]))))
        if energy > best:
            best = energy
            best_at = start
    return clip[best_at : best_at + window]


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(_unit(left), _unit(right)))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm
