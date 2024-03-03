import logging
from pathlib import Path
from typing import Callable, Dict, Generator, List, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float32
from torch._tensor import Tensor
from torch.utils.data import IterableDataset, default_collate, get_worker_info

from dynamic_fusion.utils.dataset import discretized_events_to_tensors
from dynamic_fusion.utils.datatypes import Batch, CropDefinition, CroppedReconstructionSample, GrayImageFloat, ReconstructionSample
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.transform import TransformDefinition

from .configuration import DatasetConfiguration, SharedConfiguration
from collections import defaultdict


class CocoIterableDataset(IterableDataset):  # type: ignore
    config: DatasetConfiguration
    shared_config: SharedConfiguration
    directory_list: List[Path]
    augmentation: Callable[[ReconstructionSample, float], CroppedReconstructionSample]
    logger: logging.Logger
    scales: Dict[int, float]

    def __init__(
        self,
        augmentation: Callable[[ReconstructionSample, float], CroppedReconstructionSample],
        config: DatasetConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.directory_list = [path for path in config.dataset_directory.glob("**/*") if path.is_dir()]
        self.config = config
        self.shared_config = shared_config
        self.augmentation = augmentation
        self.logger = logging.getLogger("CocoDataset")
        self.scales = defaultdict(float)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError("__getitem__ not implemented for IterableDataset!")

    def reset_scale_in_current_worker(self) -> None:
        worker_id = get_current_worker_id()
        self.scales[worker_id] = np.random.random() * (self.config.max_upscaling - 1) + 1
        self.logger.info(f"Setting worker {worker_id} ID to {self.scales[worker_id]}")

    def __iter__(
        self,
    ) -> Generator[
        Tuple[
            Float32[torch.Tensor, "Time SubBin X Y"],  # polarity sum
            Float32[torch.Tensor, "Time SubBin X Y"],  # mean
            Float32[torch.Tensor, "Time SubBin X Y"],  # std
            Float32[torch.Tensor, "Time SubBin X Y"],  # event count
            GrayImageFloat,
            TransformDefinition,
            CropDefinition,
        ],
        None,
        None,
    ]:
        while True:
            index = np.random.randint(0, len(self.directory_list))
            name = "downscaled_discretized_events" if self.shared_config.spatial_upscaling else "discretized_events"
            threshold_path = self.directory_list[index] / f"{name}_{self.config.threshold}.h5"
            with h5py.File(threshold_path, "r") as file:
                discretized_events = DiscretizedEvents.load_from_file(file)

            input_path = self.directory_list[index] / "input.h5"
            with h5py.File(input_path, "r") as file:
                preprocessed_image: GrayImageFloat = np.array(file["preprocessed_image"])
                transform_definition = TransformDefinition.load_from_file(file)

            event_polarity_sum, timestamp_mean, timestamp_std, event_count = discretized_events_to_tensors(discretized_events)

            network_data = ReconstructionSample(event_polarity_sum, timestamp_mean, timestamp_std, event_count, preprocessed_image)
            worker_id = get_current_worker_id()
            if self.scales[worker_id] == 0:
                self.reset_scale_in_current_worker()

            for _ in range(self.config.augmentation_tries):
                try:
                    augmented_network_data = self.augmentation(network_data, self.scales[worker_id])
                except ValueError as ex:
                    self.logger.warning(f"Encountered error {ex} when trying to transform {self.directory_list[index]}, retrying transforms.")
                    continue

                if self._validate(augmented_network_data.sample):
                    yield (
                        augmented_network_data.sample.event_polarity_sums,
                        augmented_network_data.sample.timestamp_means,
                        augmented_network_data.sample.timestamp_stds,
                        augmented_network_data.sample.event_counts,
                        preprocessed_image,
                        transform_definition,
                        augmented_network_data.crop_definition,
                    )
                    break
            else:  # This happens if no break
                self.logger.warning(f"No valid data found for dir {self.directory_list[index].name}, skipping!")

    def _validate(self, sample: ReconstructionSample) -> bool:
        if torch.all(sample.event_counts == 0):
            return False

        max_of_mean_polarities_over_times = einops.reduce((sample.event_polarity_sums != 0).to(torch.float32), "Time D X Y -> Time", "mean").max()

        if max_of_mean_polarities_over_times < self.shared_config.min_allowed_max_of_mean_polarities_over_times:
            return False

        return True


def collate_items_and_update_scale(
    items: List[
        Tuple[
            Float32[torch.Tensor, "Time SubBin X Y"],  # polarity sum
            Float32[torch.Tensor, "Time SubBin X Y"],  # mean
            Float32[torch.Tensor, "Time SubBin X Y"],  # std
            Float32[torch.Tensor, "Time SubBin X Y"],  # event count
            GrayImageFloat,
            TransformDefinition,
            CropDefinition,
        ],
    ],
    dataset: CocoIterableDataset,
) -> Batch:
    tensors = default_collate([tuple(x[i] for i in range(4)) for x in items])
    collated_items = (*tensors, *[[x[i] for x in items] for i in range(4, 7)])
    dataset.reset_scale_in_current_worker()
    return collated_items


def get_current_worker_id() -> int:
    return get_worker_info().id if get_worker_info() is not None else 0  # type: ignore
