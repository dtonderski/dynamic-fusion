from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandera as pa
import torch
from jaxtyping import Bool, Float, Float32, Int, Int64, Shaped, UInt8
from pandera.typing import DataFrame, Series
from typing_extensions import TypeAlias

from dynamic_fusion.utils.transform import TransformDefinition

Image: TypeAlias = UInt8[np.ndarray, "H W 3"]
GrayImage: TypeAlias = UInt8[np.ndarray, "H W"]
GrayImageFloat: TypeAlias = Float[np.ndarray, "H W"]
GrayVideoInt: TypeAlias = UInt8[np.ndarray, "T H W"]
GrayVideoFloat: TypeAlias = Float[np.ndarray, "T H W"]
GrayVideoTorch: TypeAlias = Float[torch.Tensor, "T H W"]

SegmentationMask: TypeAlias = UInt8[np.ndarray, "H W"]


class EventSchema(pa.DataFrameModel):
    timestamp: Series[np.float64]  # [s]
    x: Series[np.int64]
    y: Series[np.int64]
    polarity: Series[bool]


Events: TypeAlias = DataFrame[EventSchema]
TimeStamps: TypeAlias = Float[torch.Tensor, " N"]
Indices: TypeAlias = Int[torch.Tensor, " N"]
Polarities: TypeAlias = Bool[torch.Tensor, " N"]

EventTensors = Tuple[TimeStamps, Indices, Indices, Polarities]

SensorState: TypeAlias = Dict[str, Any]
VideoSensorData: TypeAlias = Tuple[List[Events], List[GrayImage]]

DiscretizedEventsStatistics: TypeAlias = Float[torch.Tensor, "T D H W"]

EventMask: TypeAlias = Bool[torch.Tensor, " N"]
TemporalBinIndices: TypeAlias = Int64[torch.Tensor, " N"]
TemporalSubBinIndices: TypeAlias = Int64[torch.Tensor, " N*D"]


class Checkpoint(TypedDict):
    encoding_state_dict: Optional[Dict[str, Any]]
    optimizer_state_dict: Optional[Dict[str, Any]]
    decoding_state_dict: Optional[Dict[str, Any]]
    iteration: Optional[int]


@dataclass
class CropDefinition:
    x_start: int = 0
    y_start: int = 0
    T_start: int = 0
    x_size: int = 0
    y_size: int = 0
    # total_number_of_bins is for convenience, used to calculate time
    # for continuous time training
    total_number_of_bins: int = 0
    placeholder: bool = False

    def crop_spatial(self, tensor: Shaped[np.ndarray, "... X Y"]) -> Shaped[np.ndarray, "... XCropped YCropped"]:
        if self.placeholder:
            return tensor
        return tensor[..., self.x_start : self.x_start + self.x_size, self.y_start : self.y_start + self.y_size]


Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # polarity sum
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # mean
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # std
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # event count
    Float32[torch.Tensor, "Batch Time 1 X Y"],  # bin end frame, unused in implicit
    List[GrayImageFloat],
    List[TransformDefinition],
    List[CropDefinition],
]

TestBatch: TypeAlias = Tuple[
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # polarity sum
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # mean
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # std
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # event count
    List[Float32[torch.Tensor, "Time SubBin XDownscaled YDownscaled"]],  # polarity sum
    List[Float32[torch.Tensor, "Time SubBin XDownscaled YDownscaled"]],  # mean
    List[Float32[torch.Tensor, "Time SubBin XDownscaled YDownscaled"]],  # std
    List[Float32[torch.Tensor, "Time SubBin XDownscaled YDownscaled"]],  # event count
    List[GrayImageFloat],
    List[TransformDefinition],
]


@dataclass
class ReconstructionSample:
    event_polarity_sums: Float32[torch.Tensor, "Time SubBin X Y"]
    timestamp_means: Float32[torch.Tensor, "Time SubBin X Y"]
    timestamp_stds: Float32[torch.Tensor, "Time SubBin X Y"]
    event_counts: Float32[torch.Tensor, "Time SubBin X Y"]
    video: Float32[torch.Tensor, "Time 1 X Y"]


@dataclass
class CroppedReconstructionSample:
    sample: ReconstructionSample
    crop_definition: CropDefinition
