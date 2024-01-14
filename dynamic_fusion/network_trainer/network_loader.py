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
        encoding_network, decoding_network = self._load_networks()
        return encoding_network, decoding_network

    def _load_networks(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        total_input_shape = self.config.encoding.input_size * (
            1
            + self.shared_config.use_mean
            + self.shared_config.use_std
            + self.shared_config.use_count
        )

        encoding_network = ConvGruNetV1(
            input_size=total_input_shape,
            hidden_size=self.config.encoding.hidden_size,
            out_size=self.config.encoding.output_size,
            kernel_size=self.config.encoding.kernel_size,
        )

        if self.config.encoding_checkpoint_path:
            checkpoint = torch.load(self.config.encoding_checkpoint_path)
            # For backward compatibility (key was changed)
            if checkpoint["encoding_state_dict"]:
                encoding_network.load_state_dict(checkpoint["encoding_state_dict"])
            # For compatibility reasons
            elif checkpoint["reconstruction_state_dict"]:  # type: ignore
                encoding_network.load_state_dict(
                    checkpoint["reconstruction_state_dict"]  # type: ignore
                )

        if not self.shared_config.implicit:
            return encoding_network, None

        input_shape = self.config.encoding.output_size
        if self.shared_config.feature_unfolding:
            input_shape *= 9
        input_shape += 1

        decoding_network = MLP(
            input_size=input_shape,
            hidden_size=self.config.decoding.hidden_size,
            hidden_layers=self.config.decoding.hidden_layers,
        )

        if self.config.decoding_checkpoint_path:
            checkpoint = torch.load(self.config.decoding_checkpoint_path)
            decoding_network.load_state_dict(checkpoint["decoding_state_dict"])

        return encoding_network, decoding_network
