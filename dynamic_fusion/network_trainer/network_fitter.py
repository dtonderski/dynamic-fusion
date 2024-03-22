import logging
import time
from typing import Iterator, Optional

import einops
import numpy as np
import torch
import torch.jit
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from dynamic_fusion.utils.dataset import CocoTestDataset, get_ground_truth, get_initial_aps_frames
from dynamic_fusion.utils.datatypes import Batch
from dynamic_fusion.utils.loss import UncertaintyLoss, get_reconstruction_loss, get_uncertainty_loss
from dynamic_fusion.utils.network import (
    accumulate_gradients,
    apply_gradients,
    network_data_to_device,
    run_decoder,
    run_decoder_with_spatial_upscaling,
    stack_and_maybe_unfold_c_list,
    to_numpy,
)
from dynamic_fusion.utils.superresolution import get_upscaling_pixel_indices_and_distances

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
        if self.shared_config.predict_uncertainty:
            self.reconstruction_loss_function = get_uncertainty_loss(self.config.reconstruction_loss_name, self.device)
        else:
            self.reconstruction_loss_function = get_reconstruction_loss(self.config.reconstruction_loss_name, self.device)
        self.logger = logging.getLogger("NetworkFitter")
        self.monitor = monitor

    def run(
        self,
        data_loader: DataLoader,  # type: ignore
        test_dataset: CocoTestDataset,
        encoding_network: nn.Module,
        decoding_network: Optional[nn.Module],
    ) -> None:
        params = list(encoding_network.parameters())
        if decoding_network is not None:
            params += list(decoding_network.parameters())

        optimizer = Adam(params, lr=self.config.lr_reconstruction)

        start_iteration = self.monitor.initialize(test_dataset, encoding_network, optimizer, decoding_network)

        encoding_network.to(self.device)
        if decoding_network is not None:
            decoding_network.to(self.device)

        data_loader_iterator = iter(data_loader)

        encoding_gradients = None
        decoding_gradients = None

        for iteration in range(start_iteration, self.config.number_of_training_iterations):
            if self.shared_config.spatial_upscaling:
                assert decoding_network is not None, "Need decoding network for spatial upsampling!"
                self._reconstruction_step_with_spatial_upscaling(data_loader_iterator, encoding_network, optimizer, decoding_network, iteration)
            else:
                self._reconstruction_step(data_loader_iterator, encoding_network, optimizer, decoding_network, iteration)

            encoding_gradients = accumulate_gradients(encoding_network, encoding_gradients)
            if decoding_network:
                decoding_gradients = accumulate_gradients(decoding_network, decoding_gradients)

            if self.config.gradient_application_period > 0 and iteration % self.config.gradient_application_period == 0:
                apply_gradients(encoding_network, encoding_gradients)
                if decoding_network:
                    apply_gradients(decoding_network, decoding_gradients)  # type: ignore
                encoding_gradients = None
                decoding_gradients = None
                optimizer.step()

            self.monitor.on_iteration(iteration, encoding_network, decoding_network)

            if iteration % self.config.network_saving_frequency == 0:
                self.monitor.save_checkpoint(iteration, encoding_network, optimizer, decoding_network)

    def _reconstruction_step(
        self,
        data_loader_iterator: Iterator[Batch],
        encoding_network: nn.Module,
        optimizer: Optimizer,
        decoding_network: Optional[nn.Module],
        iteration: int,
    ) -> None:
        """Variable name explanations:
            1. T - time in video, in [0, 101] for the default data generation with 100 bins per sequence
            2. tau - time in frame, in [0,1].
            3. t - time in current sequence, in [0, sequence_length]. Note that T = tau + t + crop.T_start
            4. c - output of encoding network
            5. y - ground truth
            6. r - output of decoding network

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
            (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, preprocessed_images, transforms, crops) = network_data_to_device(
                next(data_loader_iterator), self.device, self.shared_config.use_mean, self.shared_config.use_std, self.shared_config.use_count
            )

        batch_size = len(preprocessed_images)
        image_loss = torch.tensor(0.0).to(event_polarity_sums)

        forward_start = time.time()

        visualize = iteration % self.config.visualization_frequency == 0
        event_polarity_sum_list, images, reconstructions = [], [], []
        c_list = []

        first_aps_frames = get_initial_aps_frames(preprocessed_images, transforms, crops, True, self.device)
        current_frame_info = None

        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            event_polarity_sum = event_polarity_sums[:, t]
            if self.shared_config.feed_initial_aps_frame:
                current_frame_info = first_aps_frames if t == 0 else torch.zeros_like(first_aps_frames)

            # TODO: fix variable names. Non-trivial because this might be output video
            # or latent codes
            c_t = encoding_network(
                event_polarity_sum,
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
                current_frame_info,
            )

            if decoding_network is None:
                raise NotImplementedError()
                # image_loss += self.reconstruction_loss_function(c_t, video[:, t, ...]).mean()  # pylint: disable=not-callable
                # if not visualize:
                #     continue
                # event_polarity_sum_list.append(to_numpy(event_polarity_sum))
                # images.append(to_numpy(video[:, t, ...]))
                # reconstructions.append(to_numpy(c_t))
                # continue

            c_list.append(c_t.clone())
        if decoding_network is None:
            return

        # Unfold c
        cs = stack_and_maybe_unfold_c_list(c_list, self.shared_config.spatial_unfolding)

        # Sample tau
        taus = np.random.rand(batch_size, self.shared_config.sequence_length)  # B T

        # Generate ground truth for taus
        gt = get_ground_truth(taus, preprocessed_images, transforms, crops, True, self.device)
        gt = einops.rearrange(gt, "B T X Y -> T B 1 X Y")

        # Calculate start and end index to use for calculating loss
        t_start = self.config.skip_first_timesteps + self.shared_config.temporal_unfolding

        # Calculate loss
        taus = einops.repeat(torch.tensor(taus).to(cs), "B T -> T B X Y 1", X=gt.shape[-2], Y=gt.shape[-1])

        for t, r_t in run_decoder(decoding_network, cs, taus, self.shared_config.temporal_interpolation, self.shared_config.temporal_unfolding, t_start):
            r_t = einops.rearrange(r_t, "B X Y C -> B C X Y")
            image_loss = image_loss + self.reconstruction_loss_function(r_t, gt[t]).mean()  # pylint: disable=not-callable

            if visualize:
                event_polarity_sum_list.append(to_numpy(event_polarity_sums[:, t]))
                images.append(to_numpy(gt[t]))
                reconstructions.append(to_numpy(r_t[:, 0:1]))

        image_loss /= cs.shape[0] - t_start
        time_forward = time.time() - forward_start

        with Timer() as timer_backward:
            image_loss.backward()  # type: ignore

        time_batch, time_backward = timer_batch.interval, timer_backward.interval
        self.logger.info(f"Iteration: {iteration}, times: {time_batch=:.2f}, {time_forward=:.2f}, {time_backward=:.2f}, {image_loss=:.3f} (reconstruction)")

        self.monitor.on_reconstruction(image_loss.item(), iteration)
        if visualize:
            self.monitor.visualize(
                np.stack(event_polarity_sum_list, 1),
                np.stack(images, 1),
                np.stack(reconstructions, 1),
                iteration,
                encoding_network,
                decoding_network,
            )

    def _reconstruction_step_with_spatial_upscaling(
        self, data_loader_iterator: Iterator[Batch], encoding_network: nn.Module, optimizer: Optimizer, decoder: nn.Module, iteration: int
    ) -> None:
        """Variable name explanations:
            1. T - time in video, in [0, 101] for the default data generation with 100 bins per sequence
            2. tau - time in frame, in [0,1].
            3. t - time in current sequence, in [0, sequence_length]. Note that T = tau + t + crop.T_start
            4. c - output of encoding network
            5. y - ground truth
            6. r - output of decoding network

        Args:
            data_loader_iterator (Iterator[Batch]): _description_
            encoding_network (nn.Module): _description_
            optimizer (Optimizer): _description_
            decoding_network (nn.Module): _description_
            iteration (int): _description_

        Raises:
            ValueError: _description_
        """
        optimizer.zero_grad()
        encoding_network.reset_states()

        temporal_interpolation, temporal_unfolding = self.shared_config.temporal_interpolation, self.shared_config.temporal_unfolding

        # region Load data, define used regions in downscaled and upscaled images, and validate that they have enough events
        batch_start = time.time()

        (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, preprocessed_images, transforms, crops) = network_data_to_device(
            next(data_loader_iterator), self.device, self.shared_config.use_mean, self.shared_config.use_std, self.shared_config.use_count
        )

        nearest_pixels, start_to_end_vectors = get_upscaling_pixel_indices_and_distances(tuple(event_counts.shape[-2:]), tuple(crops[0].grid.shape[-3:-1]))

        time_batch = time.time() - batch_start
        # endregion

        # region Forward
        forward_start = time.time()
        visualize = iteration % self.config.visualization_frequency == 0
        event_polarity_sum_list, images, reconstructions = [], [], []
        c_list = []

        first_aps_frames = get_initial_aps_frames(preprocessed_images, transforms, crops, False, self.device)
        current_frame_info = None

        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            if self.shared_config.feed_initial_aps_frame:
                current_frame_info = first_aps_frames if t == 0 else torch.zeros_like(first_aps_frames)

            c_t = encoding_network(
                event_polarity_sums[:, t],
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
                current_frame_info,
            )

            c_list.append(c_t.clone())

        # Unfold c
        cs = stack_and_maybe_unfold_c_list(c_list, self.shared_config.spatial_unfolding)

        # Sample tau
        taus = np.random.rand(len(preprocessed_images), self.shared_config.sequence_length)  # B T

        # Generate ground truth for taus
        gt = get_ground_truth(taus, preprocessed_images, transforms, crops, False, self.device)
        gt = einops.rearrange(gt, "B T X Y -> T B 1 X Y")

        # Calculate start and end index to use for calculating loss
        t_start = self.config.skip_first_timesteps + temporal_unfolding

        # Calculate loss
        image_loss = torch.tensor(0.0).to(event_polarity_sums)
        taus = einops.rearrange(torch.tensor(taus).to(cs), "B T -> T B")

        for t, r_t in run_decoder_with_spatial_upscaling(decoder, cs, taus, temporal_interpolation, temporal_unfolding, nearest_pixels, start_to_end_vectors, t_start):
            image_loss = image_loss + self.reconstruction_loss_function(r_t, gt[t]).mean()

            if visualize:
                event_polarity_sum_list.append(to_numpy(event_polarity_sums[:, t]))
                images.append(to_numpy(gt[t]))
                reconstructions.append(to_numpy(r_t))

        image_loss /= cs.shape[0] - t_start
        time_forward = time.time() - forward_start
        # endregion

        with Timer() as timer_backward:
            image_loss.backward()  # type: ignore

        time_backward = timer_backward.interval
        self.logger.info(f"Iteration: {iteration}, times: {time_batch=:.2f}, {time_forward=:.2f}, {time_backward=:.2f}, {image_loss=:.3f} (reconstruction)")

        self.monitor.on_reconstruction(image_loss.item(), iteration)
        if visualize:
            self.monitor.visualize_upsampling(
                np.stack(event_polarity_sum_list, 1),
                np.stack(images, 1),
                np.stack(reconstructions, 1),
                iteration,
                encoding_network,
                decoder,
            )
            return
