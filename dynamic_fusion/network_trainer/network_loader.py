import logging
from typing import Optional, Tuple

import torch
from torch import nn

from dynamic_fusion.networks.decoding_nets.mlp import MLP
from dynamic_fusion.networks.reconstruction_nets import ConvGruNetV1
from dynamic_fusion.networks.reconstruction_nets.e2vid_net import E2VIDRecurrent

from .configuration import NetworkLoaderConfiguration, SharedConfiguration


class NetworkLoader:
    config: NetworkLoaderConfiguration
    shared_config: SharedConfiguration
    logger: logging.Logger

    def __init__(self, config: NetworkLoaderConfiguration, shared_config: SharedConfiguration) -> None:
        self.config = config
        self.shared_config = shared_config
        self.logger = logging.getLogger("DataGenerator")

    def run(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        encoding_network, decoding_network = self._load_networks()
        return encoding_network, decoding_network

    def _load_networks(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        total_input_size = 0
        if self.shared_config.use_events:
            total_input_size = self.config.encoding.input_size * (1 + self.shared_config.use_mean + self.shared_config.use_std + self.shared_config.use_count)
        if self.shared_config.use_aps_for_all_frames:
            total_input_size += 2
        elif self.shared_config.use_initial_aps_frame:
            total_input_size += 1

        if total_input_size == 0:
            raise ValueError("We must use either events or APS data!")
        if total_input_size == 1:
            self.logger.warning("Only initial APS frame used!")

        if self.shared_config.implicit:
            output_size = self.config.encoding.output_size
        elif self.shared_config.predict_uncertainty:
            output_size = 2
        else:
            output_size = 1

        encoding_network: nn.Module
        if self.config.encoding.architecture == "ConvGruNetV1":
            encoding_network = ConvGruNetV1(
                input_size=total_input_size,
                hidden_size=self.config.encoding.hidden_size,
                out_size=output_size,
                kernel_size=self.config.encoding.kernel_size,
                use_time_to_prev_ev=self.shared_config.use_events,
                old_norm=self.config.old_norm
            )
        elif self.config.encoding.architecture == "E2VID":
            encoding_network = E2VIDRecurrent(input_size=total_input_size, output_size=output_size)

        if self.config.encoding_checkpoint_path:
            checkpoint = torch.load(self.config.encoding_checkpoint_path)  # type: ignore
            # For backward compatibility (key was changed)
            if "encoding_state_dict" in checkpoint.keys() and checkpoint["encoding_state_dict"]:
                encoding_network.load_state_dict(checkpoint["encoding_state_dict"])
            # For compatibility reasons
            elif "reconstruction_state_dict" in checkpoint.keys() and checkpoint["reconstruction_state_dict"]:
                encoding_network.load_state_dict(checkpoint["reconstruction_state_dict"])

        if not self.shared_config.implicit:
            return encoding_network, None

        input_shape = self.config.encoding.output_size
        if self.shared_config.spatial_unfolding:
            input_shape *= 9
        if self.shared_config.temporal_unfolding:
            input_shape *= 3
        input_shape += 1
        if self.shared_config.spatial_upscaling:
            input_shape += 2

        decoding_network = MLP(
            input_size=input_shape,
            hidden_size=self.config.decoding.hidden_size,
            hidden_layers=self.config.decoding.hidden_layers,
            output_size=2 if self.shared_config.predict_uncertainty else 1,
        )

        if self.config.decoding_checkpoint_path:
            checkpoint = torch.load(self.config.decoding_checkpoint_path)  # type: ignore
            decoding_network.load_state_dict(checkpoint["decoding_state_dict"])

        return encoding_network, decoding_network
