from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import einops
import numpy as np
import pandera as pa
import torch
from jaxtyping import Bool, Float, Float32, Int, Int64, Shaped, UInt8
from pandera.typing import DataFrame, Series
from torch.nn.functional import grid_sample
from typing_extensions import TypeAlias

from dynamic_fusion.utils.transform import TransformDefinition

Image: TypeAlias = UInt8[np.ndarray, "X Y 3"]
GrayImage: TypeAlias = UInt8[np.ndarray, "X Y"]
GrayImageFloat: TypeAlias = Float[np.ndarray, "X Y"]
GrayVideoInt: TypeAlias = UInt8[np.ndarray, "T X Y"]
GrayVideoFloat: TypeAlias = Float[np.ndarray, "T X Y"]
GrayVideoTorch: TypeAlias = Float[torch.Tensor, "T X Y"]

SegmentationMask: TypeAlias = UInt8[np.ndarray, "X Y"]


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

DiscretizedEventsStatistics: TypeAlias = Float[torch.Tensor, "T D X Y"]

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
    T_start: int
    T_end: int
    total_number_of_bins: int
    # This defines how to generate the UPSCALED ground truth
    grid: Float32[torch.Tensor, "1 X Y 2"]
    # This defines how the input is cropped
    x_start: Optional[int] = None
    x_stop: Optional[int] = None
    y_start: Optional[int] = None
    y_stop: Optional[int] = None

    def crop_output_spatial(self, video: Shaped[torch.Tensor, "T X Y"]) -> Shaped[torch.Tensor, "T X Y"]:
        grid_repeated = einops.repeat(self.grid, "1 X Y N -> T X Y N", T=video.shape[0])
        video_expanded = einops.rearrange(video, "T X Y -> T 1 X Y")
        sampled = grid_sample(video_expanded, grid_repeated, mode="bicubic", align_corners=True)
        return sampled[:, 0]

    def crop_input_spatial(self, video: Shaped[torch.Tensor, "T X Y"]) -> Shaped[torch.Tensor, "T X Y"]:
        if (self.x_start is None or self.y_start is None or self.x_stop is None or self.y_stop is None):
            raise ValueError("Input cropping is not defined for this CropDefinition!")
        return video[:, self.x_start : self.x_stop, self.y_start : self.y_stop]


Batch: TypeAlias = Tuple[
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # polarity sum
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # mean
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # std
    Float32[torch.Tensor, "Batch Time SubBin X Y"],  # event count
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
    preprocessed_image: GrayImageFloat
    transform_definition: TransformDefinition


@dataclass
class CroppedReconstructionSample:
    sample: ReconstructionSample
    crop_definition: CropDefinition
