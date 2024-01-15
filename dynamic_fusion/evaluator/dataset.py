import logging
from pathlib import Path
from typing import List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float, Float32
from torch.utils.data import Dataset

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


class CocoTestDataset(Dataset):  # type: ignore
    threshold: float
    implicit: bool
    timesteps_in_bin: List[float]
    use_random_timesteps: int = False
    timestamps: Float[torch.Tensor, " NDataset TMax"]
    directory_list: List[Path]
    logger: logging.Logger

    def __init__(
        self,
        dataset_directory: Path,
        implicit: bool,
        timestamps_in_bin: Optional[List[float]],
        threshold: float = 1.4,
        maximum_sequence_length: int = 150,
        data_generator_target_image_size: Optional[Tuple[int, int]] = None
    ) -> None:
        self.logger = logging.getLogger("CocoDataset")
        self.threshold = threshold
        self.implicit = implicit
        self.data_generator_target_image_size = data_generator_target_image_size

        self.directory_list = [
            path for path in dataset_directory.glob("**/*") if path.is_dir()
        ]
        if timestamps_in_bin is None:
            self.timesteps_in_bin = [1]
            if implicit:
                self.use_random_timesteps = True
                self.timestamps = torch.rand(len(self), maximum_sequence_length)
        else:
            self.timesteps_in_bin = timestamps_in_bin

    def __len__(self) -> int:
        return len(self.directory_list)

    def __getitem__(self, index: int) -> Tuple[
        Float32[torch.Tensor, "Time SubBin X Y"],  # polarity sum
        Float32[torch.Tensor, "Time SubBin X Y"],  # mean
        Float32[torch.Tensor, "Time SubBin X Y"],  # std
        Float32[torch.Tensor, "Time SubBin X Y"],  # event count
        Float32[torch.Tensor, "Time 1 X Y"],  # bin end frame, unused in implicit
        Optional[Float32[torch.Tensor, "N Time"]],  # continuous timestamps
        Optional[Float32[torch.Tensor, "N Time 1 X Y"]],  # frames at timestamps
    ]:
        directory = self.directory_list[index]

        threshold_path = directory / f"discretized_events_{self.threshold}.h5"
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
            network_data.video.shape[2],
            network_data.video.shape[3],
            network_data.video.shape[0],
            network_data.video.shape[0],
        )

        if self.implicit or self.timesteps_in_bin != [1]:
            if self.use_random_timesteps:
                timestamps_in_bins: Float[torch.Tensor, " 1 T"] = self.timestamps[
                    index : index + 1, : video.shape[0]
                ]
            else:
                timestamps: Float[torch.Tensor, " N"] = torch.tensor(
                    self.timesteps_in_bin
                )
                timestamps_in_bins = einops.repeat(
                    timestamps,
                    "N -> N T",
                    T=network_data.video.shape[0],
                )

            videos_at_timestamps = []
            for timestamps_in_bins_i in timestamps_in_bins:
                videos_at_timestamps.append(
                    generate_frames_at_continuous_timestamps(
                        timestamps_in_bins_i,
                        preprocessed_image,
                        transform_definition,
                        crop_definition,
                        self.data_generator_target_image_size
                    )
                )
            
            video_at_continuous_timestamps = torch.stack(videos_at_timestamps)
        else:
            timestamps_in_bins = torch.zeros(1)
            video_at_continuous_timestamps = torch.zeros(1)

        return (
            network_data.event_polarity_sums,
            network_data.timestamp_means,
            network_data.timestamp_stds,
            network_data.event_counts,
            network_data.video,
            timestamps_in_bins,
            video_at_continuous_timestamps,
        )
