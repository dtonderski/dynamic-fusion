import numpy as np
from dynamic_fusion.utils.datatypes import Batch


import torch


def network_data_to_device(
    batch: Batch,
    device: torch.device,
    use_mean: bool,
    use_std: bool,
    use_count: bool,
) -> Batch:
    (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, video, preprocessed_image, transforms_definition, crop_definition) = batch
    event_polarity_sums = event_polarity_sums.to(device)

    if use_mean:
        timestamp_means = timestamp_means.to(device)
    if use_std:
        timestamp_stds = timestamp_stds.to(device)
    if use_count:
        event_counts = event_counts.to(device)

    video = video.to(device)

    return (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, video, preprocessed_image, transforms_definition, crop_definition)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:  # type: ignore[type-arg]
    return tensor.detach().cpu().numpy()  # type: ignore[no-any-return]
