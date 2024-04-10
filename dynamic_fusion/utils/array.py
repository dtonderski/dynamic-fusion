from typing import Union

import numpy as np
import torch


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:  # type: ignore[type-arg]
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()  # type: ignore[no-any-return]
