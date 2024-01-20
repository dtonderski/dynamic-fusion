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
        continuous_timestamps_in_bins
        + np.arange(0, continuous_timestamps_in_bins.shape[0])
        + crop_definition.T_start
    )

    # Now, translate this to video time, knowing the total number of bins in the video
    # If we have 2 bins, then their timestamps are currently (0,1), (1,2), and
    # should be mapped to (0, 0.5), (0.5, 1). Therefore, this is trivial
    continuous_timestamps_using_video_time = (
        continuous_timestamps_using_bin_time / crop_definition.total_number_of_bins
    )

    timestamps_and_zero = torch.concat(
        [torch.zeros(1), continuous_timestamps_using_video_time]
    )

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
