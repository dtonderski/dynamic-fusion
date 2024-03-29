from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from dynamic_fusion.network_trainer.configuration import NetworkLoaderConfiguration


class NetworkHandlerConfiguration(BaseModel):
    use_mean: bool = Field(...)
    use_std: bool = Field(...)
    use_count: bool = Field(...)
    implicit: bool = Field(...)
    spatial_unfolding: bool = Field(...)
    temporal_unfolding: bool = Field(...)
    spatial_upscaling: bool = Field(...)
    data_generator_target_image_size: Optional[Tuple[int, int]] = Field(...)
    losses: List[str] = Field(...)


class VisualizerConfiguration(BaseModel):
    network_handler: NetworkHandlerConfiguration = Field(...)
    network_loader: NetworkLoaderConfiguration = Field(...)
    total_bins_in_video: int = Field(...)  # TODO: should be settable in the UI or even loaded from data
