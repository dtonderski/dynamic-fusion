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
from dynamic_fusion.utils.dataset import generate_frames_at_continuous_timestamps

from dynamic_fusion.utils.datatypes import Batch
from dynamic_fusion.utils.loss import get_reconstruction_loss
from dynamic_fusion.utils.network import network_data_to_device, to_numpy
from dynamic_fusion.utils.video import get_video

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
        self.reconstruction_loss_function = get_reconstruction_loss(self.config.reconstruction_loss_name, self.device)
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

        for iteration in range(start_iteration, self.config.number_of_training_iterations):
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
        """Variable name explanations:
            1. T - timestamp in video, in [0, 101] for the default data generation with 100 bins per image
            2. tau - timestamp in frame, in [0,1].
            3. t - timestamp in current sequence, in [0, sequence_length]. Note that T = tau + t + crop.T_start

        Args:
            data_loader_iterator (Iterator[Batch]): _description_
            encoding_network (nn.Module): _description_
            optimizer (Optimizer): _description_
            decoding_network (Optional[nn.Module]): _description_
            iteration (int): _description_

        Raises:
            ValueError: _description_
        """
        optimizer.zero_grad()
        encoding_network.reset_states()

        with Timer() as timer_batch:
            (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, video, preprocessed_images, transforms, crops) = (
                network_data_to_device(
                    next(data_loader_iterator),
                    self.device,
                    self.shared_config.use_mean,
                    self.shared_config.use_std,
                    self.shared_config.use_count,
                )
            )

        image_loss = torch.tensor(0.0).to(event_polarity_sums)

        forward_start = time.time()

        visualize = iteration % self.config.visualization_frequency == 0
        event_polarity_sum_list, images, reconstructions = [], [], []
        c_list = []

        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            event_polarity_sum = event_polarity_sums[:, t]

            # TODO: fix variable names. Non-trivial because this might be output video
            # or latent codes
            c_t = encoding_network(
                event_polarity_sum,
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
            )

            if decoding_network is None:
                image_loss += self.reconstruction_loss_function(c_t, video[:, t, ...]).mean()  # pylint: disable=not-callable
                if not visualize:
                    continue
                event_polarity_sum_list.append(to_numpy(event_polarity_sum))
                images.append(to_numpy(video[:, t, ...]))
                reconstructions.append(to_numpy(c_t))
                continue

            c_list.append(c_t.clone())
        if decoding_network is None:
            return
        t_0 = self.config.skip_first_timesteps + self.shared_config.temporal_unfolding
        t_end = self.shared_config.sequence_length - self.shared_config.temporal_interpolation - self.shared_config.temporal_unfolding
        cs = torch.concat(c_list, dim=2)  # B C T X Y

        if self.shared_config.spatial_unfolding:
            cs = torch.nn.functional.unfold(cs, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1)
            cs = einops.rearrange(cs, " B C (T X Y) -> B C T X Y", T=self.shared_config.sequence_length, X=event_polarity_sum.shape[-2])
        if self.shared_config.temporal_unfolding:
            cs = torch.nn.functional.unfold(cs, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1)
            cs = einops.rearrange(cs, " B C (T X Y) -> B C T X Y", T=self.shared_config.sequence_length, X=event_polarity_sum.shape[-2])

        cs = einops.rearrange(cs, "B C T X Y -> T B C X Y")

        T_starts = np.array([x.T_start for x in crops])
        taus = np.random.rand(len(preprocessed_images), t_end - t_0)
        Ts = einops.rearrange(np.arange(t_0, t_end), "T -> 1 T") + taus + T_starts
        Ts_normalized = Ts / crops[0].total_number_of_bins  # Normalize from [0,NBinsInSequence] to [0,1]
        xs_list = []
        for image, transform, T_normalized_batch in zip(preprocessed_images, transforms, Ts_normalized):
            xs_list.append(get_video(image, transform, T_normalized_batch, self.config.data_generator_target_image_size, self.device))
        xs = einops.rearrange(torch.concat(xs_list, dim=0), "B T X Y -> T B 1 X Y")

        taus = einops.rearrange(taus, "B T -> T B 1 X Y")

        for t in range(t_0, t_end):
            r_t = decoding_network(torch.concat([cs[t], taus[t]], dim=1))  # type: ignore
            if self.shared_config.temporal_interpolation:
                r_tnext = decoding_network(torch.concat([cs[t + 1], taus[t] - 1], dim=1))  # type: ignore
                r_t = r_t * (1 - taus[t]) + r_tnext * (taus[t])
            image_loss = image_loss + self.reconstruction_loss_function(r_t, xs[t]).mean()  # pylint: disable=not-callable

            if not visualize:
                continue

            event_polarity_sum_list.append(to_numpy(event_polarity_sum))
            images.append(to_numpy(xs[:, t, ...]))
            reconstructions.append(to_numpy(r_t))

        image_loss /= self.shared_config.sequence_length
        time_forward = time.time() - forward_start

        with Timer() as timer_backward:
            image_loss.backward()  # type: ignore
            optimizer.step()

        time_batch, time_backward = timer_batch.interval, timer_backward.interval
        self.logger.info(
            f"Iteration: {iteration}, times: {time_batch=:.2f}, {time_forward=:.2f}, {time_backward=:.2f}, {image_loss=:.3f} (reconstruction)"
        )

        self.monitor.on_reconstruction(image_loss.item(), iteration)
        if visualize:
            self.monitor.visualize(
                np.stack(event_polarity_sum_list, 1), np.stack(images, 1), np.stack(reconstructions, 1), iteration, cs[-8:], decoding_network  # type: ignore
            )
