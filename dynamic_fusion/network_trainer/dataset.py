import logging
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float32
from torch._tensor import Tensor
from torch.utils.data import Dataset, IterableDataset

from dynamic_fusion.utils.discretized_events import DiscretizedEvents

from .configuration import DatasetConfiguration, SharedConfiguration
from .utils.datatypes import ReconstructionSample


class CocoIterableDataset(IterableDataset):  # type: ignore
    config: DatasetConfiguration
    directory_list: List[Path]
    transform: Optional[Callable[[ReconstructionSample], ReconstructionSample]]
    logger: logging.Logger

    def __init__(
        self,
        config: DatasetConfiguration,
        transform: Optional[Callable[[ReconstructionSample], ReconstructionSample]],
    ) -> None:
        self.directory_list = [
            path for path in config.dataset_directory.glob("**/*") if path.is_dir()
        ]
        self.config = config
        self.transform = transform
        self.logger = logging.getLogger("CocoDataset")

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError("__getitem__ not implemented for IterableDataset!")

    def __iter__(
        self,
    ) -> Generator[
        Tuple[
            Float32[torch.Tensor, "Time Threshold X Y"],
            Float32[torch.Tensor, "Time Threshold X Y"],
            Float32[torch.Tensor, "Time Threshold X Y"],
            Float32[torch.Tensor, "Time Threshold X Y"],
            Float32[torch.Tensor, "Time 1 X Y"],
        ],
        None,
        None,
    ]:
        while True:
            # TODO: add continuous time here
            # Reconstruction sample should also include times of frames in bin
            # Generating it before cropping the tensor in time is inefficient, but it shouldn't
            # matter with num_workers = 4 - batch generation should not be a bottleneck. 
            # to be verified, though. Can use len(discretized_events) to calculate
            # timestamps I think. First, generate len(discretized_events) samples from U[0,1], 
            # then get the size of each bin by 1/len(discretized_events) 
            index = np.random.randint(0, len(self.directory_list))

            threshold_path = (
                self.directory_list[index] / f"discretized_events_{self.config.threshold}.h5"
            )
            with h5py.File(threshold_path, "r") as file:
                discretized_events = DiscretizedEvents.load_from_file(file)

            video_path = self.directory_list[index] / "ground_truth.h5"
            with h5py.File(video_path, "r") as file:
                video = torch.from_numpy(np.array(file["synchronized_video"])).to(
                    torch.float32
                )

            event_polarity_sum, timestamp_mean, timestamp_std, event_count = (
                self._discretized_events_to_tensors(discretized_events)
            )

            network_data = ReconstructionSample(
                event_polarity_sum,
                timestamp_mean,
                timestamp_std,
                event_count,
                einops.rearrange(video, "Time X Y -> Time 1 X Y"),
            )
            for _ in range(self.config.transform_tries):
                if self.transform:
                    try:
                        network_data = self.transform(network_data)
                    except ValueError as ex:
                        self.logger.warning(
                            f"Encountered error {ex} when trying to transform"
                            f" {self.directory_list[index]}, retrying transforms."
                        )

                if self._validate(network_data):
                    yield (
                        network_data.event_polarity_sums,
                        network_data.timestamp_means,
                        network_data.timestamp_stds,
                        network_data.event_counts,
                        network_data.video,
                    )
                    break
            else:  # This happens if no break
                self.logger.warning(
                    f"No valid data found for dir {self.directory_list[index].name},"
                    " skipping!"
                )

    def _discretized_events_to_tensors(
        self,
        discretized_events: DiscretizedEvents,
    ) -> Tuple[
        Float32[torch.Tensor, "Time 1 X Y"],
        Float32[torch.Tensor, "Time 1 X Y"],
        Float32[torch.Tensor, "Time 1 X Y"],
        Float32[torch.Tensor, "Time 1 X Y"],
    ]:
        return (
            discretized_events.event_polarity_sum.to(torch.float32),
            discretized_events.timestamp_mean.to(torch.float32),
            discretized_events.timestamp_std.to(torch.float32),
            discretized_events.event_count.to(torch.float32),
        )

    def _validate(self, sample: ReconstructionSample) -> bool:
        if torch.all(sample.event_counts == 0):
            return False

        max_of_mean_polarities_over_times = einops.reduce(
            (sample.event_polarity_sums != 0).to(torch.float32),
            "Time D X Y -> Time",
            "mean",
        ).max()

        if (
            max_of_mean_polarities_over_times
            < self.config.min_allowed_max_of_mean_polarities_over_times
        ):
            return False

        return True
