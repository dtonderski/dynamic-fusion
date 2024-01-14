from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from dynamic_fusion.networks.decoding_nets.mlp import MLP

from dynamic_fusion.networks.reconstruction_nets import ConvGruNetV1

from .configuration import NetworkLoaderConfiguration, SharedConfiguration


class NetworkLoader:
    config: NetworkLoaderConfiguration
    shared_config: SharedConfiguration

    def __init__(
        self, config: NetworkLoaderConfiguration, shared_config: SharedConfiguration
    ) -> None:
        self.config = config
        self.shared_config = shared_config

    def run(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        reconstruction_network, decoding_network = self._load_reconstruction_network()
        return reconstruction_network, decoding_network

    def _load_reconstruction_network(self) -> Tuple[nn.Module, Optional[nn.Module]]:

        total_input_shape = self.config.reconstruction.input_size * (
            1
            + self.shared_config.use_mean
            + self.shared_config.use_std
            + self.shared_config.use_count
        )

        reconstruction_network = ConvGruNetV1(
            input_size=total_input_shape,
            hidden_size=self.config.reconstruction.hidden_size,
            out_size=self.config.reconstruction.output_size,
            kernel_size=self.config.reconstruction.kernel_size,
        )
        
        if self.config.reconstruction_checkpoint_path:
            checkpoint = torch.load(self.config.reconstruction_checkpoint_path)
            reconstruction_network.load_state_dict(
                checkpoint["reconstruction_state_dict"]
            )

        if not self.shared_config.implicit:
            return reconstruction_network, None

        decoding_network = MLP(
            input_size=self.config.decoding.input_size,
            hidden_size=self.config.decoding.hidden_size,
            hidden_layers=self.config.decoding.hidden_layers,
        )

        if self.config.decoding_checkpoint_path:
            checkpoint = torch.load(self.config.decoding_checkpoint_path)
            decoding_network.load_state_dict(checkpoint["decoding_state_dict"])

        return reconstruction_network, decoding_network
