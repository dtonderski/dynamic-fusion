import numpy as np
import torch
from jaxtyping import Float32
from torch.utils.data import DataLoader, Dataset
from dynamic_fusion.utils.datatypes import (
    CropDefinition,
    CroppedReconstructionSample,
    ReconstructionSample,
)

from dynamic_fusion.utils.image import scale_video_to_quantiles

from .configuration import (
    DataHandlerConfiguration,
    SharedConfiguration,
    AugmentationConfiguration,
)
from .dataset import CocoIterableDataset


class CocoAugmentation:
    config: AugmentationConfiguration
    shared_config: SharedConfiguration

    def __init__(self, config: AugmentationConfiguration, shared_config: SharedConfiguration):
        self.config = config
        self.shared_config = shared_config

    def __call__(self, network_data: ReconstructionSample) -> CroppedReconstructionSample:
        # network_data.video = scale_video_to_quantiles(network_data.video)
        transformed_network_data = self._crop_tensors(network_data)
        return transformed_network_data

    def extract_part_of_tensor(
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

    def _crop_tensors(self, network_data: ReconstructionSample) -> CroppedReconstructionSample:
        max_x_start = network_data.video.shape[2] - self.config.network_image_size[0]
        max_y_start = network_data.video.shape[3] - self.config.network_image_size[1]

        x_start, y_start = np.random.randint(low=0, high=(max_x_start, max_y_start))
        total_number_of_bins = network_data.video.shape[0]

        t_start = np.random.randint(
            low=0,
            high=total_number_of_bins - self.shared_config.sequence_length + 1,  # + 1 because exclusive
        )

        network_data.event_polarity_sums = self.extract_part_of_tensor(network_data.event_polarity_sums, t_start, x_start, y_start)

        if self.shared_config.use_mean:
            network_data.timestamp_means = self.extract_part_of_tensor(network_data.timestamp_means, t_start, x_start, y_start)

        if self.shared_config.use_std:
            network_data.timestamp_stds = self.extract_part_of_tensor(network_data.timestamp_stds, t_start, x_start, y_start)

        if self.shared_config.use_count:
            network_data.event_counts = self.extract_part_of_tensor(network_data.event_counts, t_start, x_start, y_start)

        network_data.video = network_data.video[
            t_start : t_start + self.shared_config.sequence_length,
            :,
            x_start : x_start + self.config.network_image_size[0],
            y_start : y_start + self.config.network_image_size[1],
        ]

        crop_definition = CropDefinition(
            x_start,
            y_start,
            t_start,
            *self.config.network_image_size,
            self.shared_config.sequence_length,
            total_number_of_bins,
        )

        return CroppedReconstructionSample(network_data, crop_definition)


class DataHandler:
    config: DataHandlerConfiguration
    shared_config: SharedConfiguration
    dataset: Dataset  # type: ignore

    def __init__(self, config: DataHandlerConfiguration, shared_config: SharedConfiguration) -> None:
        self.config = config
        self.shared_config = shared_config
        augmentation = CocoAugmentation(config.augmentation, shared_config)
        self.dataset = CocoIterableDataset(augmentation, config.dataset, shared_config)

    def run(self) -> DataLoader:  # type: ignore
        return DataLoader(self.dataset, self.config.batch_size, num_workers=self.config.num_workers)
