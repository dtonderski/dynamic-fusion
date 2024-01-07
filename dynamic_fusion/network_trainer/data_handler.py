import numpy as np
import torch
from jaxtyping import Float32
from torch.utils.data import DataLoader, Dataset

from dynamic_fusion.utils.image import scale_video_to_quantiles

from .configuration import (
    DataHandlerConfiguration,
    SharedConfiguration,
    TransformsConfiguration,
)
from .dataset import CocoIterableDataset
from .utils.datatypes import ReconstructionSample, TransformedReconstructionSample


class CocoTransform:
    config: TransformsConfiguration
    shared_config: SharedConfiguration

    def __init__(
        self, config: TransformsConfiguration, shared_config: SharedConfiguration
    ):
        self.config = config
        self.shared_config = shared_config

    def __call__(
        self, network_data: ReconstructionSample
    ) -> TransformedReconstructionSample:
        network_data.video = scale_video_to_quantiles(network_data.video)
        transformed_network_data = self._crop_tensors(network_data)
        return transformed_network_data

    def _crop_tensors(
        self, network_data: ReconstructionSample
    ) -> TransformedReconstructionSample:
        max_x_start = network_data.video.shape[2] - self.config.network_image_size[0]
        max_y_start = network_data.video.shape[3] - self.config.network_image_size[1]

        x_start, y_start = np.random.randint(low=0, high=(max_x_start, max_y_start))
        total_video_length = network_data.video.shape[0]
        t_start = np.random.randint(
            low=0,
            high=total_video_length - self.shared_config.sequence_length,
        )

        network_data.event_polarity_sums = self._extract_part_of_tensor(
            network_data.event_polarity_sums, t_start, x_start, y_start
        )

        if self.shared_config.use_mean:
            network_data.timestamp_means = self._extract_part_of_tensor(
                network_data.timestamp_means, t_start, x_start, y_start
            )

        if self.shared_config.use_std:
            network_data.timestamp_stds = self._extract_part_of_tensor(
                network_data.timestamp_stds, t_start, x_start, y_start
            )

        if self.shared_config.use_count:
            network_data.event_counts = self._extract_part_of_tensor(
                network_data.event_counts, t_start, x_start, y_start
            )

        network_data.video = network_data.video[
            t_start : t_start + self.shared_config.sequence_length,
            :,
            x_start : x_start + self.config.network_image_size[0],
            y_start : y_start + self.config.network_image_size[1],
        ]

        return TransformedReconstructionSample(
            network_data, x_start, y_start, t_start, total_video_length
        )

    def _extract_part_of_tensor(
        self,
        tensor: Float32[torch.Tensor, "Time SubBin X Y"],
        t_start: int,
        x_start: int,
        y_start: int,
    ) -> Float32[torch.Tensor, "Time SubBin X Y"]:
        return tensor[
            t_start : t_start + self.shared_config.sequence_length,
            :,
            x_start : x_start + self.config.network_image_size[0],
            y_start : y_start + self.config.network_image_size[1],
        ]


class DataHandler:
    config: DataHandlerConfiguration
    shared_config: SharedConfiguration
    dataset: Dataset  # type: ignore

    def __init__(
        self, config: DataHandlerConfiguration, shared_config: SharedConfiguration
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        transform = CocoTransform(config.transform, shared_config)
        self.dataset = CocoIterableDataset(config.dataset, transform)

    def run(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.dataset, self.config.batch_size, num_workers=self.config.num_workers
        )
