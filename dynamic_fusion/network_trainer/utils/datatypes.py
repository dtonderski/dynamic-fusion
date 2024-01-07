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

@dataclass
class CropDefinition:
    x_start: int
    y_start: int
    t_start: int
    x_size: int
    y_size: int
    t_size: int
    # total_number_of_bins is for convenience, used to calculate time
    # for continuous time training
    total_number_of_bins: int

@dataclass
class CroppedReconstructionSample:
    sample: ReconstructionSample
    transformation: CropDefinition


Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "batch Time SubBin X Y"],  # EPS
    Float32[torch.Tensor, "batch Time SubBin X Y"],  # Means
    Float32[torch.Tensor, "batch Time SubBin X Y"],  # STD
    Float32[torch.Tensor, "batch Time SubBin X Y"],  # Counts
    Float32[torch.Tensor, "batch Time 1 X Y"],  # Video
]
