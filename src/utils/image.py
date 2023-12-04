from typing import List

import numpy as np
import torch
from jaxtyping import Float32


def scale_video_to_quantiles(
    video: Float32[torch.Tensor, "Time X Y"],
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> Float32[torch.Tensor, "Time X Y"]:
    video_low_quantile, video_high_quantile = torch.quantile(
        video, torch.tensor([low_quantile, high_quantile])
    )
    clipped_video = torch.clip(video, min=video_low_quantile, max=video_high_quantile)
    value_range = torch.maximum(
        video_high_quantile - video_low_quantile, torch.tensor(1e-5)
    )
    clipped_video = (clipped_video - video_low_quantile) / value_range
    return video


def scale_to_quantiles_numpy(
    img: np.ndarray,  # type: ignore[type-arg]
    axis: List[int],
    q_low: float = 0.0,
    q_high: float = 1.0,
) -> np.ndarray:  # type: ignore[type-arg]
    dtype = img.dtype
    qtls = np.quantile(img, [q_low, q_high], axis=axis, keepdims=True)
    img = np.clip(img, qtls[0, ...], qtls[1, ...])
    den = np.maximum(qtls[1, ...] - qtls[0, ...], 1e-5)
    img = (img - qtls[0, ...]) / den
    return img.astype(dtype)
