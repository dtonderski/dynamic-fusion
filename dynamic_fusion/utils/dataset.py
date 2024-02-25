from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float32

from dynamic_fusion.utils.datatypes import CropDefinition, GrayImageFloat
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.transform import TransformDefinition
from dynamic_fusion.utils.video import get_video


def discretized_events_to_tensors(
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


def generate_frames_at_continuous_timestamps(
    continuous_timestamps_in_bins: Float32[np.ndarray, " T"],
    preprocessed_image: GrayImageFloat,
    transform_definition: TransformDefinition,
    crop_definition: CropDefinition,
    data_generator_target_image_size: Optional[Tuple[int, int]] = None,
) -> Float32[torch.Tensor, "T 1 X Y"]:
    # Translate from time in bin to time in video
    # For example, if continuous time in bin is 0.5 (domain is [0, 1]), it's bin number 2, and t_start is 1,
    # then the result will be 3.5.
    continuous_timestamps_using_bin_time = (
        continuous_timestamps_in_bins + np.arange(0, continuous_timestamps_in_bins.shape[0]) + crop_definition.T_start
    )

    # Now, translate this to video time, knowing the total number of bins in the video
    # If we have 2 bins, then their timestamps are currently (0,1), (1,2), and
    # should be mapped to (0, 0.5), (0.5, 1). Therefore, this is trivial
    continuous_timestamps_using_video_time = continuous_timestamps_using_bin_time / crop_definition.total_number_of_bins

    timestamps_and_zero = torch.concat([torch.zeros(1), continuous_timestamps_using_video_time])

    frames_and_zero = get_video(
        preprocessed_image,
        transform_definition,
        timestamps_and_zero,
        data_generator_target_image_size,
        device=torch.device("cuda"),
    )

    cropped_frames = frames_and_zero[
        1:,
        crop_definition.x_start : crop_definition.x_start + crop_definition.x_size,
        crop_definition.y_start : crop_definition.y_start + crop_definition.y_size,
    ]

    return einops.rearrange(cropped_frames, "Time X Y -> Time 1 X Y")


class CocoTestDataset(Dataset):  # type: ignore
    directory_list: List[Path]
    threshold: float
    logger: logging.Logger

    def __init__(self, dataset_directory: Path, threshold: float = 1.4) -> None:
        self.directory_list = sorted([path for path in dataset_directory.glob("**/*") if path.is_dir()])
        self.logger = logging.getLogger("CocoDataset")
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.directory_list)

    def __getitem__(self, index: int) -> Tuple[
        Float[torch.Tensor, "Time SubBin X Y"],  # polarity sum
        Float[torch.Tensor, "Time SubBin X Y"],  # mean
        Float[torch.Tensor, "Time SubBin X Y"],  # std
        Float[torch.Tensor, "Time SubBin X Y"],  # event count
        Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # polarity sum
        Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # mean
        Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # std
        Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # event count
        GrayImageFloat,
        TransformDefinition,
    ]:
        with h5py.File(self.directory_list[index] / f"discretized_events_{self.threshold}.h5", "r") as file:
            discretized_events = DiscretizedEvents.load_from_file(file)

        with h5py.File(self.directory_list[index] / f"downscaled_discretized_events_{self.threshold}.h5", "r") as file:
            downscaled_discretized_events = DiscretizedEvents.load_from_file(file)

        input_path = self.directory_list[index] / "input.h5"
        with h5py.File(input_path, "r") as file:
            preprocessed_image: GrayImageFloat = np.array(file["preprocessed_image"])
            transform_definition = TransformDefinition.load_from_file(file)

        return (
            *discretized_events_to_tensors(discretized_events),
            *discretized_events_to_tensors(downscaled_discretized_events),
            preprocessed_image,
            transform_definition,
        )


def collate_test_items(
    items: List[
        Tuple[
            Float[torch.Tensor, "Time SubBin X Y"],  # polarity sum
            Float[torch.Tensor, "Time SubBin X Y"],  # mean
            Float[torch.Tensor, "Time SubBin X Y"],  # std
            Float[torch.Tensor, "Time SubBin X Y"],  # event count
            Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # polarity sum
            Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # mean
            Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # std
            Float[torch.Tensor, "Time SubBin XDownscaled YDownscaled"],  # event count
            GrayImageFloat,
            TransformDefinition,
        ],
    ],
) -> TestBatch:
    tensors = default_collate([tuple(x[i] for i in range(4)) for x in items])
    collated_items = (*tensors, *[[x[i] for x in items] for i in range(4, 10)])
    return collated_items


def get_ground_truth(
    taus: Float[torch.Tensor, "B T"],
    preprocessed_images: List[GrayImageFloat],
    transforms: List[TransformDefinition],
    device: torch.device,
    crops: Optional[List[CropDefinition]],
    data_generator_target_image_size: Optional[Tuple[int, int]],
) -> Float[torch.Tensor, "B T X Y"]:
    T_starts = einops.rearrange(np.array([crop.T_start for crop in crops]), "B -> B 1")
    Ts = einops.rearrange(np.arange(taus.shape[1]), "T -> 1 T") + taus + T_starts
    Ts_normalized_batch = Ts / crops[0].total_number_of_bins  # Normalize from [0,sequence_length] to [0,1]
    ys_list = []
    for i, (image, transform, Ts_normalized) in enumerate(zip(preprocessed_images, transforms, Ts_normalized_batch)):
        video_batch = get_video(image, transform, Ts_normalized, data_generator_target_image_size, device)
        ys_list.append(crops[i].crop_spatial(video_batch) if crops is not None else video_batch)

    return torch.stack(ys_list, dim=0)
