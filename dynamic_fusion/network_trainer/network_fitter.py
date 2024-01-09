import logging
import time
from typing import Iterator
import einops

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from .configuration import NetworkFitterConfiguration, SharedConfiguration
from .training_monitor import TrainingMonitor
from .utils.datatypes import Batch
from .utils.loss import LPIPS
from .utils.network import (
    network_data_to_device,
    to_numpy,
)
from .utils.timer import Timer


class NetworkFitter:
    config: NetworkFitterConfiguration
    shared_config: SharedConfiguration
    reconstruction_loss_function: nn.Module
    logger: logging.Logger
    monitor: TrainingMonitor
    device: torch.device

    def __init__(
        self,
        monitor: TrainingMonitor,
        config: NetworkFitterConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_loss_function = self._get_reconstruction_loss()
        self.logger = logging.getLogger("NetworkFitter")
        self.monitor = monitor

    def run(
        self,
        data_loader: DataLoader,  # type: ignore
        reconstruction_network: nn.Module,
        decoding_network: nn.Module,
    ) -> None:
        params = list(reconstruction_network.parameters()) + list(
            decoding_network.parameters()
        )
        optimizer = Adam(params, lr=self.config.lr_reconstruction)

        start_iteration = self.monitor.initialize(
            data_loader,
            reconstruction_network,
            optimizer,
            decoding_network,
        )

        reconstruction_network.to(self.device)
        decoding_network.to(self.device)

        data_loader_iterator = iter(data_loader)

        for iteration in range(
            start_iteration, self.config.number_of_training_iterations
        ):
            self._reconstruction_step(
                data_loader_iterator,
                reconstruction_network,
                optimizer,
                decoding_network,
                iteration,
            )

            if iteration % self.config.network_saving_frequency == 0:
                self.monitor.save_checkpoint(
                    reconstruction_network,
                    optimizer,
                    decoding_network,
                    iteration,
                )

    def _reconstruction_step(
        self,
        data_loader_iterator: Iterator[Batch],
        reconstruction_network: nn.Module,
        optimizer: Optimizer,
        decoding_network: nn.Module,
        iteration: int,
    ) -> None:
        optimizer.zero_grad()
        reconstruction_network.reset_states()

        with Timer() as timer_batch:
            (
                event_polarity_sums,
                timestamp_means,
                timestamp_stds,
                event_counts,
                video,
                continuous_timestamps,
                continuous_timestamp_frames,
            ) = network_data_to_device(
                next(data_loader_iterator),
                self.device,
                self.shared_config.use_mean,
                self.shared_config.use_std,
                self.shared_config.use_count,
            )

        image_loss = torch.tensor(0.0).to(event_polarity_sums)

        forward_start = time.time()

        visualize = iteration % self.config.visualization_frequency == 0
        event_polarity_sum_list, images, predictions = [], [], []

        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            event_polarity_sum = event_polarity_sums[:, t]

            # TODO: fix variable names. Non-trivial because this might be output video
            # or latent codes
            prediction = reconstruction_network(
                event_polarity_sum,
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
            )

            if t >= self.config.skip_first_timesteps:
                # TODO: if using continuous timestamps
                if True:
                    expanded_timestamps = continuous_timestamps[:, t, None, None, None].expand_as(
                        continuous_timestamp_frames[:, t]
                    )  # B 1 X Y
                    encoding_and_time = torch.concatenate(
                        [prediction, expanded_timestamps], dim=1
                    )

                    encoding_and_time = einops.rearrange(encoding_and_time, "B C X Y -> B X Y C")
                    decoding_prediction = decoding_network(encoding_and_time)
                    prediction = einops.rearrange(
                        decoding_prediction, "B X Y 1 -> B 1 X Y"
                    )

                image_loss += (
                    self.reconstruction_loss_function(  # pylint: disable=not-callable
                        prediction, continuous_timestamp_frames[:, t, ...]
                    ).mean()
                )

                if visualize:
                    event_polarity_sum_list.append(to_numpy(event_polarity_sum))
                    images.append(to_numpy(continuous_timestamp_frames[:, t, ...]))
                    predictions.append(to_numpy(prediction))
        image_loss /= self.shared_config.sequence_length
        time_forward = time.time() - forward_start

        with Timer() as timer_backward:
            image_loss.backward()  # type: ignore
            optimizer.step()

        time_batch, time_backward = timer_batch.interval, timer_backward.interval
        self.logger.info(
            f"Iteration: {iteration}, times: {time_batch=:.2f}, {time_forward=:.2f},"
            f" {time_backward=:.2f}, {image_loss=:.3f} (reconstruction)"
        )

        self.monitor.on_reconstruction(image_loss.item(), iteration)
        if visualize:
            self.monitor.visualize(
                np.stack(event_polarity_sum_list, 1),
                np.stack(images, 1),
                np.stack(predictions, 1),
                iteration,
                True,
            )

    def _get_reconstruction_loss(self) -> nn.Module:
        if self.config.reconstruction_loss_name.upper() == "L1":
            return nn.L1Loss().to(self.device)
        elif self.config.reconstruction_loss_name.upper() == "L2":
            return nn.MSELoss().to(self.device)
        elif self.config.reconstruction_loss_name.upper() == "LPIPS":
            return LPIPS().to(self.device)
        else:
            raise ValueError(
                f"Unknown image loss name: {self.config.reconstruction_loss_name}"
            )
