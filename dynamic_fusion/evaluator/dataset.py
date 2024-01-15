import logging
from pathlib import Path
from typing import List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float, Float32

from dynamic_fusion.network_trainer.dataset import CocoIterableDataset
from dynamic_fusion.utils.dataset import (
    discretized_events_to_tensors,
    generate_frames_at_continuous_timestamps,
)
from dynamic_fusion.utils.datatypes import (
    CropDefinition,
    GrayImageFloat,
    ReconstructionSample,
)
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.image import scale_video_to_quantiles
from dynamic_fusion.utils.transform import TransformDefinition

from .configuration import DatasetConfiguration, SharedConfiguration


class CocoTestDataset(Dataset):  # type: ignore
    config: DatasetConfiguration
    shared_config: SharedConfiguration
    timesteps_in_bin: List[int]
    use_random_timesteps: int
    timestamps: Float[torch.Tensor, " N TMax"]
    directory_list: List[Path]
    logger: logging.Logger

    def __init__(
        self,
        timestamps_in_bin: Optional[List[int]],
        maximum_allowed_sequence_length: int,
        config: DatasetConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.logger = logging.getLogger("CocoDataset")
        self.config = config
        self.shared_config = shared_config

        self.directory_list = [
            path for path in config.dataset_directory.glob("**/*") if path.is_dir()
        ]
        if timestamps_in_bin is None:
            self.timesteps_in_bin = [1]
            if self.shared_config.implicit:
                self.use_random_timesteps = True
                self.timestamps = torch.rand(
                    len(self), maximum_allowed_sequence_length
                )
        else:
            self.timesteps_in_bin = timestamps_in_bin

    def __len__(self) -> int:
        return len(self.directory_list) * len(self.timesteps_in_bin)

    def __getitem__(self, index: int) -> Tuple[
        Float32[torch.Tensor, "Time SubBin X Y"],  # polarity sum
        Float32[torch.Tensor, "Time SubBin X Y"],  # mean
        Float32[torch.Tensor, "Time SubBin X Y"],  # std
        Float32[torch.Tensor, "Time SubBin X Y"],  # event count
        Float32[torch.Tensor, "Time 1 X Y"],  # bin end frame, unused in implicit
        Optional[Float32[torch.Tensor, "Time"]],  # continuous timestamps
        Optional[Float32[torch.Tensor, "Time 1 X Y"]],  # frames at timestamps
    ]:
        file_index = index // len(self.timesteps_in_bin)
        directory = self.directory_list[file_index]

        threshold_path = directory / f"discretized_events_{self.config.threshold}.h5"
        with h5py.File(threshold_path, "r") as file:
            discretized_events = DiscretizedEvents.load_from_file(file)

        video_path = directory / "ground_truth.h5"
        with h5py.File(video_path, "r") as file:
            video = torch.from_numpy(np.array(file["synchronized_video"])).to(
                torch.float32
            )

        input_path = directory / "input.h5"
        with h5py.File(input_path, "r") as file:
            preprocessed_image: GrayImageFloat = np.array(file["preprocessed_image"])
            transform_definition = TransformDefinition.load_from_file(file)

        event_polarity_sum, timestamp_mean, timestamp_std, event_count = (
            discretized_events_to_tensors(discretized_events)
        )
        network_data = ReconstructionSample(
            event_polarity_sum,
            timestamp_mean,
            timestamp_std,
            event_count,
            einops.rearrange(video, "Time X Y -> Time 1 X Y"),
        )
        network_data.video = scale_video_to_quantiles(network_data.video)

        crop_definition = CropDefinition(
            0,
            0,
            0,
            network_data.video.shape[1],
            network_data.video.shape[2],
            network_data.video.shape[0],
            network_data.video.shape[0],
        )

        if self.shared_config.implicit:
            if self.use_random_timesteps:
                timestep_index = index % len(self.timesteps_in_bin)
                timestamp = torch.tensor(self.timesteps_in_bin[timestep_index])
                continuous_timestamps_in_bins = einops.repeat(
                    timestamp,
                    " -> T",
                    T=network_data.video.shape[0],
                )
            else:
                continuous_timestamps_in_bins = self.timestamps[:, : video.shape[0]]

            # TODO: video frames at the timestamps
            video_at_continuous_timestamps = generate_frames_at_continuous_timestamps(
                continuous_timestamps_in_bins,
                preprocessed_image,
                transform_definition,
                crop_definition,
            )
        else:
            continuous_timestamps_in_bins = torch.zeros(1)
            video_at_continuous_timestamps = torch.zeros(1)

        return (
            network_data.event_polarity_sums,
            network_data.timestamp_means,
            network_data.timestamp_stds,
            network_data.event_counts,
            network_data.video,
            continuous_timestamps_in_bins,
            video_at_continuous_timestamps,
        )
