import os

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from diarizers import SegmentationModel

segmentation_model = SegmentationModel().from_pretrained(
    "diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn"
)

device = torch.device("cuda")

token = os.getenv("HUGGINGFACE_API_KEY")
if not token:
    raise SystemExit("Set HUGGINGFACE_API_KEY before running diarization.")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
pipeline.to(device)

segmentation_model = segmentation_model.to_pyannote_model()
pipeline._segmentation.model = segmentation_model.to(device)

with ProgressHook() as hook:
    output = pipeline("./extracted_audio.wav", hook=hook)

for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start}s and t={turn.end}s")

with open("diarization_output.rttm", "w") as rttm_file:
    output.write_rttm(rttm_file)
