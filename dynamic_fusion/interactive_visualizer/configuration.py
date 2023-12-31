from typing import Tuple
from pydantic import BaseModel, Field

from dynamic_fusion.network_trainer.configuration import NetworkLoaderConfiguration


class NetworkHandlerConfiguration(BaseModel):
    use_mean: bool = Field(...)
    use_std: bool = Field(...)
    use_count: bool = Field(...)
    data_generator_target_image_size: Tuple[int, int] = Field(...)


class VisualizerConfiguration(BaseModel):
    network_handler: NetworkHandlerConfiguration = Field(...)
    network_loader: NetworkLoaderConfiguration = Field(...)
