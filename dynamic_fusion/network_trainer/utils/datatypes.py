from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float32
from typing_extensions import TypeAlias


@dataclass
class ReconstructionSample:
    event_polarity_sums: Float32[torch.Tensor, "Time SubBin X Y"]
    timestamp_means: Float32[torch.Tensor, "Time SubBin X Y"]
    timestamp_stds: Float32[torch.Tensor, "Time SubBin X Y"]
    event_counts: Float32[torch.Tensor, "Time SubBin X Y"]
    video: Float32[torch.Tensor, "Time 1 X Y"]


Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "batch Time SubBin X Y"], # EPS
    Float32[torch.Tensor, "batch Time SubBin X Y"], # Means
    Float32[torch.Tensor, "batch Time SubBin X Y"], # STD
    Float32[torch.Tensor, "batch Time SubBin X Y"], # Counts
    Float32[torch.Tensor, "batch Time 1 X Y"],      # Video
]
