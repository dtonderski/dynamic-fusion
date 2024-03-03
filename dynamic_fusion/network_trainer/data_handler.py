from typing import Any, Tuple

import numpy as np
from torch.utils.data import DataLoader

from dynamic_fusion.utils.dataset import CocoTestDataset
from dynamic_fusion.utils.datatypes import CroppedReconstructionSample, ReconstructionSample, CropDefinition
from dynamic_fusion.utils.superresolution import get_grid

from .configuration import AugmentationConfiguration, DataHandlerConfiguration, SharedConfiguration
from .dataset import CocoIterableDataset, collate_items_and_update_scale


class CocoAugmentation:
    config: AugmentationConfiguration
    shared_config: SharedConfiguration

    def __init__(self, config: AugmentationConfiguration, shared_config: SharedConfiguration):
        self.config = config
        self.shared_config = shared_config

    def __call__(self, network_data: ReconstructionSample, scale: float = 1) -> CroppedReconstructionSample:
        # network_data.video = scale_video_to_quantiles(network_data.video)

        max_starts = tuple(dim_size - dim_target for dim_size, dim_target in zip(network_data.event_polarity_sums.shape[-2:], self.config.network_image_size))
        x_start, y_start = np.random.randint(low=0, high=(max_starts))
        x_stop, y_stop = x_start + self.config.network_image_size[0], y_start + self.config.network_image_size[1]

        # Get temporal crop in input space
        total_number_of_bins = network_data.event_polarity_sums.shape[0]
        t_start = np.random.randint(low=0, high=total_number_of_bins - self.shared_config.sequence_length + 1)  # + 1 because exclusive
        t_end = t_start + self.shared_config.sequence_length

        # Crop inputs
        event_polarity_sums = network_data.event_polarity_sums[t_start:t_end, :, x_start:x_stop, y_start:y_stop]

        if self.shared_config.use_mean:
            timestamp_means = network_data.timestamp_means[t_start:t_end, :, x_start:x_stop, y_start:y_stop]
        if self.shared_config.use_std:
            timestamp_stds = network_data.timestamp_stds[t_start:t_end, :, x_start:x_stop, y_start:y_stop]
        if self.shared_config.use_count:
            event_counts = network_data.event_counts[t_start:t_end, :, x_start:x_stop, y_start:y_stop]

        # Don't modify network_data in-place in case we fail validation
        augmented_network_data = ReconstructionSample(event_polarity_sums, timestamp_means, timestamp_stds, event_counts, network_data.preprocessed_image)
        # Get ground truth cropping (/interpolating) grid
        output_shape = tuple(int(size * scale) for size in self.config.network_image_size)
        input_shape = network_data.preprocessed_image.shape[-2:] if self.shared_config.spatial_upscaling else self.shared_config.data_generator_target_image_size

        grid = get_grid(input_shape, output_shape, ((x_start, x_stop), (y_start, y_stop)))  # type: ignore

        crop_definition = CropDefinition(t_start, t_end, total_number_of_bins, grid)

        return CroppedReconstructionSample(augmented_network_data, crop_definition)


class DataHandler:
    config: DataHandlerConfiguration
    shared_config: SharedConfiguration
    dataset: CocoIterableDataset

    def __init__(self, config: DataHandlerConfiguration, shared_config: SharedConfiguration) -> None:
        self.config = config
        self.shared_config = shared_config
        augmentation = CocoAugmentation(config.augmentation, shared_config)
        self.dataset = CocoIterableDataset(augmentation, config.dataset, shared_config)
        # self.test_dataset = CocoTestDataset(config.test_dataset_directory)

    def _collate_fn(self, x) -> Any:
        return collate_items_and_update_scale(x, self.dataset)

    def run(self) -> Tuple[DataLoader]:  # , CocoTestDataset]:  # type: ignore
        return (
            DataLoader(
                self.dataset,
                self.config.batch_size,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
            )
            # self.test_dataset,
        )
