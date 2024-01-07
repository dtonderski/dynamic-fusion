from pathlib import Path

import torch
from torch import nn

from dynamic_fusion.networks.reconstruction_nets import ConvGruNetV1

from .configuration import (
    NetworkLoaderConfiguration,
    ReconstructionNetworkConfiguration,
    SharedConfiguration,
)


class NetworkLoader:
    config: NetworkLoaderConfiguration
    shared_config: SharedConfiguration

    def __init__(
        self, config: NetworkLoaderConfiguration, shared_config: SharedConfiguration
    ) -> None:
        self.config = config
        self.shared_config = shared_config

    def run(self) -> nn.Module:
        reconstruction_network = self._load_reconstruction_network()
        return reconstruction_network

    def _load_reconstruction_network(self) -> nn.Module:
        network_config: ReconstructionNetworkConfiguration = (
            self.config.reconstruction
        )

        total_input_shape = network_config.input_size * (
            1
            + self.shared_config.use_mean
            + self.shared_config.use_std
            + self.shared_config.use_count
        )

        reconstruction_network = ConvGruNetV1(
            input_size=total_input_shape,
            hidden_size=network_config.hidden_size,
            out_size=network_config.output_size,
            kernel_size=network_config.kernel_size,
        )
        if self.config.reconstruction_checkpoint_path:
            checkpoint = torch.load(Path(self.config.reconstruction_checkpoint_path))
            reconstruction_network.load_state_dict(
                checkpoint["reconstruction_state_dict"]
            )

        return reconstruction_network
