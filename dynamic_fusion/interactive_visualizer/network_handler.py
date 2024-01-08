from pathlib import Path
from typing import Tuple
from torch import nn
import torch
from jaxtyping import Float


class NetworkHandler:
    """This class will handle the neural network. It will load the data,
    the network, and return the data needed by the visualizer
    given the directory, the start_bin, the end_bin, and the
    continuous timestamp.
    """

    encoding_network: nn.Module
    decoding_network: nn.Module
    start_bin_index: int
    end_bin_index: int

    def __init__(self) -> None:
        pass

    def get_reconstruction(self, timestamp: int) -> Float[torch.Tensor, "X Y"]:
        raise NotImplementedError()

    def get_ground_truth(self, timestamp: int) -> Float[torch.Tensor, "X Y"]:
        raise NotImplementedError()

    def get_start_and_end_images(
        self,
    ) -> Tuple[Float[torch.Tensor, "X Y"], Float[torch.Tensor, "X Y"]]:
        raise NotImplementedError()

    def set_bin_indices(self, start: int, end: int) -> None:
        self.start_bin_index = start
        self.end_bin_index = end

    def set_data_directory(self, path: Path) -> None:
        raise NotImplementedError()
