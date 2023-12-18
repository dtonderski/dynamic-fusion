import numpy as np
from jaxtyping import Shaped


def normalize(data: Shaped[np.ndarray, "..."]) -> Shaped[np.ndarray, "..."]:
    data = data - np.min(data)
    return data / np.max(data)
