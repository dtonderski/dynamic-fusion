import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
