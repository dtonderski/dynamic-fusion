from pathlib import Path
from typing import List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from dynamic_fusion.interactive_visualizer.configuration import (
    NetworkHandlerConfiguration,
)
from dynamic_fusion.network_trainer.configuration import NetworkLoaderConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.network_trainer.training_monitor import TrainingMonitor
from dynamic_fusion.utils.dataset import discretized_events_to_tensors
from dynamic_fusion.utils.datatypes import GrayImageFloat, ReconstructionSample
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.image import scale_video_to_quantiles
from dynamic_fusion.utils.loss import get_reconstruction_loss
from dynamic_fusion.utils.transform import TransformDefinition
from dynamic_fusion.utils.video import get_video


class NetworkHandler:
    """This class will handle the neural network. It will load the data,
    the network, and return the data needed by the visualizer
    given the directory, the start_bin, the end_bin, and the
    continuous timestamp.
    """

    config: NetworkHandlerConfiguration
    encoding_network: nn.Module
    decoding_network: nn.Module
    start_bin_index: int
    end_bin_index: int
    threshold: float = 1.4
    sample: ReconstructionSample
    latent_code: Float[torch.Tensor, "1 C X Y"]
    network_loader: NetworkLoader
    preprocessed_image: GrayImageFloat
    transform_definition: TransformDefinition
    last_decoding_prediction: Optional[Float[torch.Tensor, "X Y"]] = None
    last_timestamp: Optional[float] = None
    last_ground_truth: Optional[Float[torch.Tensor, "X Y"]] = None
    device: torch.device
    losses: List[nn.Module]

    def __init__(
        self,
        config: NetworkHandlerConfiguration,
        network_loader_configuration: NetworkLoaderConfiguration,
    ) -> None:
        self.config = config
        # wrong configuration, but it's OK
        self.network_loader = NetworkLoader(network_loader_configuration, config)  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoding_network, decoding_network = self.network_loader.run()
        if decoding_network is None:
            raise ValueError("Only use implicit networks!")
        self.decoding_network = decoding_network

        self.encoding_network.to(self.device)
        self.decoding_network.to(self.device)
        print("Loaded networks!")

        self.losses = [get_reconstruction_loss(x, self.device) for x in self.config.losses]

    # Public API
    def get_reconstruction(self, timestamp: float, interpolation: bool = False) -> Float[torch.Tensor, "X Y"]:
        if self.last_decoding_prediction is not None and self.last_timestamp == timestamp:
            return self.last_decoding_prediction

        # Use previous prediction if needed
        encoding = self.previous_prediction if timestamp < 0 else self.prediction
        if interpolation and self.next_prediction is not None:
            next_prediction = self.prediction if timestamp < 0 else self.next_prediction
        timestamp = timestamp + 1 if timestamp < 0 else timestamp

        with torch.no_grad():
            expanded_timestamp = torch.tensor([timestamp], device=self.device)[:, None, None, None].expand(1, 1, *encoding.shape[-2:])
            encoding_and_time = torch.concat([encoding, expanded_timestamp], dim=1)
            encoding_and_time = einops.rearrange(encoding_and_time, "1 C X Y -> 1 X Y C")
            decoding_prediction = self.decoding_network(encoding_and_time)

            if interpolation and self.next_prediction is not None:
                next_encoding_and_time = torch.concat([next_prediction, expanded_timestamp - 1], dim=1)
                next_encoding_and_time = einops.rearrange(next_encoding_and_time, "1 C X Y -> 1 X Y C")
                next_decoding_prediction = self.decoding_network(next_encoding_and_time)

                decoding_prediction = next_decoding_prediction * (timestamp) + decoding_prediction * (1 - timestamp)

            self.last_timestamp = timestamp
            self.last_decoding_prediction = torch.squeeze(decoding_prediction).cpu()

            return self.last_decoding_prediction

    def get_ground_truth(self, timestamp: float, total_bins_in_video: int = 100) -> Float[torch.Tensor, "X Y"]:
        timestamp_using_bin_time = timestamp + self.end_bin_index
        timestamp_using_video_time = timestamp_using_bin_time / total_bins_in_video

        self.last_ground_truth = get_video(
            self.preprocessed_image,
            self.transform_definition,
            np.array([0, timestamp_using_video_time]),
            self.config.data_generator_target_image_size,
            device=torch.device("cpu"),
        )[1].numpy()
        return self.last_ground_truth

    def get_start_and_end_images(
        self,
    ) -> Tuple[GrayImageFloat, GrayImageFloat]:
        if self.end_bin_index == 0:
            start = get_video(
                self.preprocessed_image,
                self.transform_definition,
                np.zeros(1),
                self.config.data_generator_target_image_size,
                device=torch.device("cpu"),
            )[0]
        else:
            start = np.array(self.sample.video[self.end_bin_index - 1, 0].cpu())

        return start, np.array(self.sample.video[self.end_bin_index, 0].cpu())

    def get_event_image(self) -> Float[torch.Tensor, "3 X Y"]:
        polarity_sums_in_bin = self.sample.event_polarity_sums[self.end_bin_index].sum(dim=0)
        colored_event_polarity_sums = TrainingMonitor.img_to_colormap(polarity_sums_in_bin.numpy(), TrainingMonitor.create_red_blue_cmap(501))
        return colored_event_polarity_sums

    def set_bin_indices(self, start: int, end: int) -> None:
        self.start_bin_index = start
        self.end_bin_index = end
        self._run_network()

    def set_data_directory(self, path: Path) -> None:
        try:
            input_path = path / "input.h5"
            with h5py.File(input_path, "r") as file:
                self.preprocessed_image: GrayImageFloat = np.array(file["preprocessed_image"])
                self.transform_definition = TransformDefinition.load_from_file(file)

            threshold_path = path / f"discretized_events_{self.threshold}.h5"
            with h5py.File(threshold_path, "r") as file:
                discretized_events = DiscretizedEvents.load_from_file(file)

            video_path = path / "ground_truth.h5"
            with h5py.File(video_path, "r") as file:
                video = torch.from_numpy(np.array(file["synchronized_video"])).to(torch.float32)

            event_polarity_sum, timestamp_mean, timestamp_std, event_count = discretized_events_to_tensors(discretized_events)

            self.sample = ReconstructionSample(
                event_polarity_sum,
                timestamp_mean,
                timestamp_std,
                event_count,
                einops.rearrange(video, "Time X Y -> Time 1 X Y"),
            )
            self.sample.video = scale_video_to_quantiles(self.sample.video)
            print("Data loading successful!")
        except Exception as e:
            print(e)

    def get_losses(self) -> List[float]:
        prediction = einops.rearrange(self.last_decoding_prediction, "X Y -> 1 1 X Y").to(self.device)
        gt_tensor = torch.tensor(self.last_ground_truth)
        gt = einops.rearrange(gt_tensor, "X Y -> 1 1 X Y").to(self.device)  # type: ignore
        return [loss(prediction, gt).item() for loss in self.losses]

    # Private
    def _run_network(self) -> None:
        with torch.no_grad():
            self.previous_prediction = None
            self.next_prediction = None
            self.last_timestamp = None
            self.last_decoding_prediction = None

            self.encoding_network.reset_states()
            polarity_sums, means, stds, counts = self._sample_to_device()
            for t in range(self.end_bin_index - self.start_bin_index + 1):
                self.prediction = self.encoding_network(
                    polarity_sums[:, t],
                    means[:, t] if self.config.use_mean else None,
                    stds[:, t] if self.config.use_std else None,
                    counts[:, t] if self.config.use_count else None,
                )

                unfolded_prediction = torch.nn.functional.unfold(self.prediction, kernel_size=3, padding=1, stride=1)
                self.prediction = einops.rearrange(
                    unfolded_prediction,
                    "B C (X Y) -> B C X Y",
                    X=polarity_sums.shape[3],
                )

                if t == self.end_bin_index - self.start_bin_index - 1:
                    self.previous_prediction = self.prediction.clone()  # type: ignore

            self.prediction = self.prediction.clone()
            try:
                self.next_prediction = self.encoding_network(
                    polarity_sums[:, t + 1],
                    means[:, t + 1] if self.config.use_mean else None,
                    stds[:, t + 1] if self.config.use_std else None,
                    counts[:, t + 1] if self.config.use_count else None,
                )

                unfolded_prediction = torch.nn.functional.unfold(self.next_prediction, kernel_size=3, padding=1, stride=1)
                self.next_prediction = einops.rearrange(
                    unfolded_prediction,
                    "B C (X Y) -> B C X Y",
                    X=polarity_sums.shape[3],
                )

            except Exception as e:
                print(e)
                return

    def _sample_to_device(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "1 T SubBins X Y"],
        Float[torch.Tensor, "1 T SubBins X Y"],
        Float[torch.Tensor, "1 T SubBins X Y"],
        Float[torch.Tensor, "1 T SubBins X Y"],
    ]:
        polarity_sums = self.sample.event_polarity_sums[self.start_bin_index : self.end_bin_index + 2].unsqueeze_(0).to(self.device)
        means = self.sample.timestamp_means[self.start_bin_index : self.end_bin_index + 2].unsqueeze_(0).to(self.device)
        stds = self.sample.timestamp_stds[self.start_bin_index : self.end_bin_index + 2].unsqueeze_(0).to(self.device)
        counts = self.sample.event_counts[self.start_bin_index : self.end_bin_index + 2].unsqueeze_(0).to(self.device)
        return polarity_sums, means, stds, counts
