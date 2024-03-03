from typing import List

import numpy as np


def scale_to_quantiles(
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
