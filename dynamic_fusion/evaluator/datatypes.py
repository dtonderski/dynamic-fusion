from typing import Tuple

import torch
from jaxtyping import Float32
from typing_extensions import TypeAlias

Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "B Time SubBin X Y"],  # EPS
    Float32[torch.Tensor, "B Time SubBin X Y"],  # Means
    Float32[torch.Tensor, "B Time SubBin X Y"],  # STD
    Float32[torch.Tensor, "B Time SubBin X Y"],  # Counts
    Float32[torch.Tensor, "B Time 1 X Y"],  # Video
    Float32[torch.Tensor, "B N Time"],  # Continuous timestamps
    Float32[torch.Tensor, "B N Time 1 X Y"],  # Continuous timestamp frames
]
