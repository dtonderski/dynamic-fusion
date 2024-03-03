from pathlib import Path
from typing import List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch
from jaxtyping import Float
from skimage.transform import resize
from torch import nn

from dynamic_fusion.interactive_visualizer.configuration import NetworkHandlerConfiguration
from dynamic_fusion.network_trainer.configuration import NetworkLoaderConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.network_trainer.training_monitor import TrainingMonitor
from dynamic_fusion.utils.dataset import discretized_events_to_tensors
from dynamic_fusion.utils.datatypes import GrayImageFloat, ReconstructionSample
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.loss import get_reconstruction_loss
from dynamic_fusion.utils.superresolution import get_spatial_upscaling_output, get_upscaling_pixel_indices_and_distances
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
    last_r: Optional[Float[torch.Tensor, "X Y"]] = None
    last_tau: Optional[float] = None
    last_used_temporal_interpolation: Optional[bool] = None
    last_ground_truth: Optional[Float[torch.Tensor, "X Y"]] = None
    last_upsampled_resolution: Optional[Tuple[int, int]] = None
    device: torch.device
    losses: List[nn.Module]
    cs: Float[torch.Tensor, "T B X Y C"]

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
    def get_reconstruction(self, tau: float, temporal_interpolation: bool = False, upsampled_resolution: Tuple[int, int] = (180, 240)) -> Float[torch.Tensor, "X Y"]:
        if self.last_r is not None and self.last_tau == tau and self.last_used_temporal_interpolation == temporal_interpolation:
            if not self.config.spatial_upscaling:
                return self.last_r
            elif self.last_upsampled_resolution == upsampled_resolution:
                return self.last_r

        if self.config.spatial_upscaling and self.last_upsampled_resolution != upsampled_resolution:
            nearest_pixels, start_to_end_vectors, out_of_bounds = get_upscaling_pixel_indices_and_distances(tuple(self.cs.shape[-3:-1]), upsampled_resolution)  # type: ignore
            self.nearest_pixels = torch.tensor(nearest_pixels)
            self.start_to_end_vectors = torch.tensor(start_to_end_vectors)

            # Define region to upsample
            # First, ensure we avoid any pixels whose upsampling required pixels outside of the bounds of the image
            self.within_bounds_mask = torch.tensor(~out_of_bounds.any(axis=0)).to(self.device)

        # Use previous prediction if needed
        t = 2
        if tau < 0:
            t = t - 1

        c = self.cs[t]
        if self.config.spatial_upscaling:
            temporal_interpolation = True
        c_next = self.cs[t + 1] if temporal_interpolation else None

        tau = tau + 1 if tau < 0 else tau

        with torch.no_grad():
            b, x, y, _ = c.shape
            tau_expanded = einops.repeat(torch.tensor([tau], device=self.device), "1 -> B X Y 1", B=b, X=x, Y=y)
            if self.config.temporal_unfolding:
                c = torch.concat([self.cs[t - 1], self.cs[t], self.cs[t + 1]], dim=-1)
                if temporal_interpolation:
                    c_next = torch.concat([self.cs[t], self.cs[t + 1], self.cs[t + 2]], dim=-1)

            if self.config.spatial_upscaling:
                r = get_spatial_upscaling_output(self.decoding_network, c, tau, c_next, self.nearest_pixels, self.start_to_end_vectors)
            else:
                r = self.decoding_network(torch.concat([c, tau_expanded], dim=-1))
                if temporal_interpolation:
                    r_next = self.decoding_network(torch.concat([c_next, tau_expanded - 1], dim=-1))
                    r = r * (1 - tau_expanded) + r_next * (tau_expanded)

            self.last_tau = tau
            self.last_r = torch.squeeze(r).cpu()

            return self.last_r

    def get_ground_truth(self, timestamp: float, total_bins_in_video: int = 100) -> Float[torch.Tensor, "X Y"]:
        timestamp_using_bin_time = timestamp + self.end_bin_index - 2
        timestamp_using_video_time = timestamp_using_bin_time / total_bins_in_video

        self.last_ground_truth = get_video(
            self.preprocessed_image,
            self.transform_definition,
            np.array([0, timestamp_using_video_time]),
            self.config.data_generator_target_image_size,
            device=torch.device("cpu"),
        )[1].numpy()
        return self.last_ground_truth

    def get_downscaled_ground_truth(self) -> Float[torch.Tensor, "XDownscaled YDownscaled"]:
        downscaled_gt = resize(self.last_ground_truth, output_shape=self.sample.event_polarity_sums.shape[-2:], order=3, anti_aliasing=True)
        return downscaled_gt

    def get_start_and_end_images(
        self,
    ) -> Tuple[GrayImageFloat, GrayImageFloat]:
        start = np.array(self.sample.video[self.end_bin_index - 3, 0].cpu())

        return start, np.array(self.sample.video[self.end_bin_index - 2, 0].cpu())

    def get_event_image(self) -> Float[torch.Tensor, "3 X Y"]:
        polarity_sums_in_bin = self.sample.event_polarity_sums[self.end_bin_index - 2].sum(dim=0)
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

            name = "downscaled_discretized_events" if self.config.spatial_upscaling else "discretized_events"
            threshold_path = path / f"{name}_{self.threshold}.h5"

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
            self.last_upsampled_resolution = None
            print("Data loading successful!")
        except Exception as e:
            print(e)

    def get_losses(self) -> List[float]:
        prediction = einops.rearrange(self.last_r, "X Y -> 1 1 X Y").to(self.device)
        gt_tensor = torch.tensor(self.last_ground_truth)
        gt = einops.rearrange(gt_tensor, "X Y -> 1 1 X Y").to(self.device)  # type: ignore
        return [loss(prediction, gt).item() for loss in self.losses]

    def get_gt_size(self) -> Tuple[int, int]:
        return tuple(self.last_ground_truth.shape[-2:])  # type: ignore

    # Private
    def _run_network(self) -> None:
        with torch.no_grad():
            self.previous_prediction = None
            self.next_prediction = None
            self.last_tau = None
            self.last_r = None

            self.encoding_network.reset_states()
            polarity_sums, means, stds, counts = self._sample_to_device()
            c_list = []

            for t in range(self.end_bin_index - self.start_bin_index + 1):
                c_list.append(
                    self.encoding_network(
                        polarity_sums[:, t],
                        means[:, t] if self.config.use_mean else None,
                        stds[:, t] if self.config.use_std else None,
                        counts[:, t] if self.config.use_count else None,
                    )
                )

            # the 5 last encodings is enough to get temporal interpolation + unfolding for both this and previous

            cs = torch.stack(c_list[-5:], dim=0)  # T B C X Y
            if self.config.spatial_unfolding:
                cs = einops.rearrange(cs, "T B C X Y -> (T B) C X Y")
                cs = torch.nn.functional.unfold(cs, kernel_size=(3, 3), padding=(1, 1), stride=1)
                self.cs = einops.rearrange(cs, "(T B) C (X Y) -> T B X Y C", T=5, X=polarity_sums.shape[-2])
            else:
                # Prepare for linear layer
                self.cs = einops.rearrange(cs, "T B C X Y -> T B X Y C")

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
