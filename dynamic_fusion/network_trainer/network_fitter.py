import logging
import time
from typing import Iterator, Optional

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from dynamic_fusion.utils.datatypes import Batch
from dynamic_fusion.utils.loss import get_reconstruction_loss
from dynamic_fusion.utils.network import network_data_to_device, to_numpy

from .configuration import NetworkFitterConfiguration, SharedConfiguration
from .training_monitor import TrainingMonitor
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
        self.reconstruction_loss_function = get_reconstruction_loss(
            self.config.reconstruction_loss_name, self.device
        )
        self.logger = logging.getLogger("NetworkFitter")
        self.monitor = monitor

    def run(
        self,
        data_loader: DataLoader,  # type: ignore
        encoding_network: nn.Module,
        decoding_network: Optional[nn.Module],
    ) -> None:
        params = list(encoding_network.parameters())
        if decoding_network is not None:
            params += list(decoding_network.parameters())

        optimizer = Adam(params, lr=self.config.lr_reconstruction)

        start_iteration = self.monitor.initialize(
            data_loader,
            encoding_network,
            optimizer,
            decoding_network,
        )

        encoding_network.to(self.device)
        if decoding_network is not None:
            decoding_network.to(self.device)

        data_loader_iterator = iter(data_loader)

        for iteration in range(
            start_iteration, self.config.number_of_training_iterations
        ):
            self._reconstruction_step(
                data_loader_iterator,
                encoding_network,
                optimizer,
                decoding_network,
                iteration,
            )

            if iteration % self.config.network_saving_frequency == 0:
                self.monitor.save_checkpoint(
                    encoding_network,
                    optimizer,
                    decoding_network,
                    iteration,
                )

    def _reconstruction_step(
        self,
        data_loader_iterator: Iterator[Batch],
        encoding_network: nn.Module,
        optimizer: Optimizer,
        decoding_network: Optional[nn.Module],
        iteration: int,
    ) -> None:
        optimizer.zero_grad()
        encoding_network.reset_states()

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
        event_polarity_sum_list, images, reconstructions = [], [], []
        previous_prediction: Optional[Float[torch.Tensor, "B C X Y"]] = None

        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            event_polarity_sum = event_polarity_sums[:, t]

            # TODO: fix variable names. Non-trivial because this might be output video
            # or latent codes
            prediction = encoding_network(
                event_polarity_sum,
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
            )

            # Store last encoding if using temporal interpolation
            if (
                t == self.config.skip_first_timesteps - 1
                and self.shared_config.implicit
                and self.shared_config.temporal_interpolation
            ):
                if self.shared_config.spatial_feature_unfolding:
                    unfolded_prediction = torch.nn.functional.unfold(
                        prediction, kernel_size=3, padding=1, stride=1
                    )
                    prediction = einops.rearrange(
                        unfolded_prediction,
                        "B C (X Y) -> B C X Y",
                        X=continuous_timestamp_frames.shape[3],
                    )

                previous_prediction = prediction.clone()

            if t < self.config.skip_first_timesteps:
                continue

            # Non-implicit
            if decoding_network is None:
                image_loss += (
                    self.reconstruction_loss_function(  # pylint: disable=not-callable
                        prediction, video[:, t, ...]
                    ).mean()
                )
                if not visualize:
                    continue
                event_polarity_sum_list.append(to_numpy(event_polarity_sum))
                images.append(to_numpy(video[:, t, ...]))
                reconstructions.append(to_numpy(prediction))
                continue

            # Implicit
            if self.shared_config.spatial_feature_unfolding:
                unfolded_prediction = torch.nn.functional.unfold(
                    prediction, kernel_size=3, padding=1, stride=1
                )
                prediction = einops.rearrange(
                    unfolded_prediction,
                    "B C (X Y) -> B C X Y",
                    X=continuous_timestamp_frames.shape[3],
                )

            expanded_timestamps = einops.repeat(
                continuous_timestamps[:, t],
                "B -> B 1 X Y",
                X=continuous_timestamp_frames.shape[3],
                Y=continuous_timestamp_frames.shape[4],
            )
            # No temporal interpolation
            if not self.shared_config.temporal_interpolation:
                encoding_and_time = torch.concat(
                    [prediction, expanded_timestamps], dim=1
                )

                encoding_and_time = einops.rearrange(
                    encoding_and_time, "B C X Y -> B X Y C"
                )

                decoding_prediction = decoding_network(encoding_and_time)
                reconstruction = einops.rearrange(
                    decoding_prediction, "B X Y 1 -> B 1 X Y"
                )

            # With temporal interpolation
            else:
                if previous_prediction is None:
                    raise ValueError("Encountered previous prediction None!")
                # expanded_timestamps in [-1, 0]
                previous_encoding_and_time = torch.concat(
                    [previous_prediction, expanded_timestamps+1], dim=1
                )
                previous_encoding_and_time = einops.rearrange(
                    previous_encoding_and_time, "B C X Y -> B X Y C"
                )

                previous_decoding_prediction = decoding_network(
                    previous_encoding_and_time
                )
                previous_decoding_prediction = einops.rearrange(
                    previous_decoding_prediction, "B X Y 1 -> B 1 X Y"
                )

                encoding_and_time = torch.concat(
                    [prediction, expanded_timestamps], dim=1
                )
                encoding_and_time = einops.rearrange(
                    encoding_and_time, "B C X Y -> B X Y C"
                )

                decoding_prediction = decoding_network(encoding_and_time)
                decoding_prediction = einops.rearrange(
                    decoding_prediction, "B X Y 1 -> B 1 X Y"
                )

                previous_prediction = prediction.clone()

                # expanded_timestamps in [-1, 0] here
                reconstruction = previous_decoding_prediction * (-expanded_timestamps)
                +(decoding_prediction * (1 + expanded_timestamps))

            image_loss += (
                self.reconstruction_loss_function(  # pylint: disable=not-callable
                    reconstruction, continuous_timestamp_frames[:, t, ...]
                ).mean()
            )

            if not visualize:
                continue

            event_polarity_sum_list.append(to_numpy(event_polarity_sum))
            images.append(to_numpy(continuous_timestamp_frames[:, t, ...]))
            reconstructions.append(to_numpy(reconstruction))

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
                np.stack(reconstructions, 1),
                iteration,
                True,
            )
