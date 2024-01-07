import logging
from pathlib import Path
from typing import Callable, Generator, List, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float32
from torch._tensor import Tensor
from torch.utils.data import IterableDataset

from dynamic_fusion.data_generator.video_generator import VideoGenerator
from dynamic_fusion.utils.datatypes import GrayImageFloat
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.transform import TransformDefinition

from .configuration import DatasetConfiguration, SharedConfiguration
from .utils.datatypes import (
    CropDefinition,
    CroppedReconstructionSample,
    ReconstructionSample,
)


class CocoIterableDataset(IterableDataset):  # type: ignore
    config: DatasetConfiguration
    shared_config: SharedConfiguration
    directory_list: List[Path]
    transform: Callable[[ReconstructionSample], CroppedReconstructionSample]
    logger: logging.Logger

    def __init__(
        self,
        transform: Callable[[ReconstructionSample], CroppedReconstructionSample],
        config: DatasetConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.directory_list = [
            path for path in config.dataset_directory.glob("**/*") if path.is_dir()
        ]
        self.config = config
        self.shared_config = shared_config
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
            Float32[torch.Tensor, "Time SubBin X Y"],  # polarity sum
            Float32[torch.Tensor, "Time SubBin X Y"],  # mean
            Float32[torch.Tensor, "Time SubBin X Y"],  # std
            Float32[torch.Tensor, "Time SubBin X Y"],  # event count
            Float32[
                torch.Tensor, "Time 1 X Y"
            ],  # frame at end of bin, unused for continuous
            Float32[torch.Tensor, "Time"],  # continuous timestamps
            Float32[torch.Tensor, "Time 1 X Y"],  # frame at continuous timestamps
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
                self.directory_list[index]
                / f"discretized_events_{self.config.threshold}.h5"
            )
            with h5py.File(threshold_path, "r") as file:
                discretized_events = DiscretizedEvents.load_from_file(file)

            video_path = self.directory_list[index] / "ground_truth.h5"
            with h5py.File(video_path, "r") as file:
                video = torch.from_numpy(np.array(file["synchronized_video"])).to(
                    torch.float32
                )

            input_path = self.directory_list[index] / "input.h5"
            with h5py.File(input_path, "r") as file:
                preprocessed_image: GrayImageFloat = np.array(
                    file["preprocessed_image"]
                )
                transform_definition = TransformDefinition.load_from_file(file)

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
                try:
                    transformed_network_data = self.transform(network_data)
                except ValueError as ex:
                    self.logger.warning(
                        f"Encountered error {ex} when trying to transform"
                        f" {self.directory_list[index]}, retrying transforms."
                    )

                if self._validate(transformed_network_data.sample):
                    continuous_timestamps_in_bins = torch.rand(
                        self.shared_config.sequence_length
                    )
                    # TODO: video frames at the timestamps
                    video_at_continuous_timestamps = (
                        self._generate_frames_at_continuous_timestamps(
                            continuous_timestamps_in_bins,
                            preprocessed_image,
                            transform_definition,
                            transformed_network_data.transformation,
                        )
                    )
                    yield (
                        network_data.event_polarity_sums,
                        network_data.timestamp_means,
                        network_data.timestamp_stds,
                        network_data.event_counts,
                        network_data.video,
                        continuous_timestamps_in_bins,
                        video_at_continuous_timestamps,
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

    def _generate_frames_at_continuous_timestamps(
        self,
        continuous_timestamps_in_bins: Float32[np.ndarray, " T"],
        preprocessed_image: GrayImageFloat,
        transform_definition: TransformDefinition,
        crop_definition: CropDefinition,
    ) -> Float32[torch.Tensor, "T 1 X Y"]:
        # Translate from time in bin to time in video
        # For example, if continuous time in bin is 0.5, it's bin number 2, and t_start is 1,
        # then the result will be 3.5.
        continuous_timestamps_using_bin_time = (
            continuous_timestamps_in_bins
            + np.arange(0, continuous_timestamps_in_bins.shape[0])
            + crop_definition.t_start
        )

        # Now, translate this to video time, knowing the total number of bins in the video
        # If we have 2 bins, then their timestamps are currently (0,1), (1,2), and
        # should be mapped to (0, 0.5), (0.5, 1). Therefore, this is trivial
        continuous_timestamps_using_video_time = (
            continuous_timestamps_using_bin_time
            / crop_definition.total_number_of_bins
        )

        frames = VideoGenerator.get_video(
            preprocessed_image,
            transform_definition,
            continuous_timestamps_using_video_time,
            (crop_definition.x_size, crop_definition.y_size),
        )

        return einops.rearrange(torch.tensor(frames), "Time X Y -> Time 1 X Y")
