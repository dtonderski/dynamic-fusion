from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandera as pa
import torch
from jaxtyping import Bool, Float, Float32, Int, Int64, UInt8
from pandera.typing import DataFrame, Series
from typing_extensions import TypeAlias

Image: TypeAlias = UInt8[np.ndarray, "H W 3"]
GrayImage: TypeAlias = UInt8[np.ndarray, "H W"]
GrayImageFloat: TypeAlias = Float[np.ndarray, "H W"]
GrayVideo: TypeAlias = UInt8[np.ndarray, "T H W"]
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

DiscretizedEventsStatistics: TypeAlias = Float[torch.Tensor, "T H W"]

EventMask: TypeAlias = Bool[torch.Tensor, " N"]
TemporalBinIndices: TypeAlias = Int64[torch.Tensor, " N"]


class Checkpoint(TypedDict):
    control_state_dict: Optional[Dict[str, Any]]
    control_optimizer_state_dict: Optional[Dict[str, Any]]
    downstream_state_dict: Optional[Dict[str, Any]]
    downstream_optimizer_state_dict: Optional[Dict[str, Any]]
    log_temperature: Optional[Float32[torch.Tensor, " 1"]]
    iteration: Optional[int]
