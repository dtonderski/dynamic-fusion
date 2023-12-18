from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float32
from typing_extensions import TypeAlias


@dataclass
class ReconstructionSample:
    event_polarity_sums: Float32[torch.Tensor, "Time 1 X Y"]
    timestamp_means: Float32[torch.Tensor, "Time 1 X Y"]
    timestamp_stds: Float32[torch.Tensor, "Time 1 X Y"]
    event_counts: Float32[torch.Tensor, "Time 1 X Y"]
    video: Float32[torch.Tensor, "Time 1 X Y"]


Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "batch Time Threshold X Y"],
    Float32[torch.Tensor, "batch Time Threshold X Y"],
    Float32[torch.Tensor, "batch Time Threshold X Y"],
    Float32[torch.Tensor, "batch Time Threshold X Y"],
    Float32[torch.Tensor, "batch Time 1 X Y"],
]
