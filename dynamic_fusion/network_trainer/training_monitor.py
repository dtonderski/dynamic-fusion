import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Tuple

import einops
import numpy as np
import skimage.transform
import torch
from jaxtyping import Float32
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.io import read_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from dynamic_fusion.utils.dataset import CocoTestDataset, collate_test_items
from dynamic_fusion.utils.datatypes import Checkpoint, TestBatch
from dynamic_fusion.utils.evaluation import MetricsDictionary, get_evaluation_image, get_evaluation_video, get_metrics, get_reconstructions_and_gt
from dynamic_fusion.utils.image import scale_to_quantiles
from dynamic_fusion.utils.network import network_test_data_to_device
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap

from .configuration import SharedConfiguration, TrainerConfiguration

LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pt"
PERSISTENT_CHECKPOINT_TEMPLATE = "checkpoint_{i}"


class TrainingMonitor:
    """This class handles checkpoints and logging"""

    training_config: TrainerConfiguration
    writer: SummaryWriter
    test_dataset: CocoTestDataset
    sample_batch: TestBatch
    device: torch.device
    shared_config: SharedConfiguration
    logger: logging.Logger
    subrun_directory: Path
    last_metrics_iteration: int = -1
    metrics: MetricsDictionary

    def __init__(self, training_config: TrainerConfiguration) -> None:
        self.training_config = training_config
        self.config = training_config.training_monitor
        self.shared_config = training_config.shared
        self.logger = logging.getLogger("TrainingMonitor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(
        self, test_dataset: CocoTestDataset, reconstruction_network: nn.Module, optimizer: torch.optim.Optimizer, decoding_network: Optional[nn.Module]
    ) -> int:
        plt.ioff()
        previous_subrun_directory, self.subrun_directory = self._get_previous_and_current_subrun_directories()
        self.subrun_directory.mkdir(parents=True, exist_ok=True)

        self._save_config()

        iteration = 0
        if self.shared_config.resume and previous_subrun_directory:
            try:
                iteration = self._load_checkpoint(
                    previous_subrun_directory,
                    reconstruction_network,
                    optimizer,
                    decoding_network,
                )
            except ValueError as ex:
                self.logger.warning(ex)

        self.writer = SummaryWriter(self.subrun_directory)  # type: ignore[no-untyped-call]
        self.test_dataset = test_dataset
        self.sample_batch = collate_test_items([test_dataset[i] for i in [0]])
        self.logger.info(f"Starting at iteration {iteration}")
        return iteration

    def _save_config(self) -> None:
        json_config = self.training_config.json(indent=4)
        with (self.subrun_directory / "config.json").open("w", encoding="utf8") as f:
            f.write(json_config)

    def _load_checkpoint(
        self,
        previous_subrun_directory: Path,
        encoding_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        decoding_network: Optional[nn.Module],
    ) -> int:
        checkpoint_path = previous_subrun_directory / LATEST_CHECKPOINT_FILENAME
        if not checkpoint_path.exists():
            raise ValueError(f"Subrun found, but no {LATEST_CHECKPOINT_FILENAME}!")

        checkpoint: Checkpoint = torch.load(checkpoint_path)  # type: ignore
        if checkpoint["encoding_state_dict"]:
            self.logger.info("Loading encoding_state_dict.")
            encoding_network.load_state_dict(checkpoint["encoding_state_dict"])
            encoding_network.to(self.device)
        # For compatibility reasons
        elif checkpoint["reconstruction_state_dict"]:  # type: ignore
            self.logger.info("Loading reconstruction_state_dict.")
            encoding_network.load_state_dict(checkpoint["reconstruction_state_dict"])  # type: ignore
            encoding_network.to(self.device)

        if decoding_network is not None and checkpoint["decoding_state_dict"]:
            self.logger.info("Loading decoding_state_dict.")
            decoding_network.load_state_dict(checkpoint["decoding_state_dict"])
            decoding_network.to(self.device)

        if checkpoint["optimizer_state_dict"]:
            self.logger.info("Loading optimizer_state_dict.")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["iteration"]:
            iteration = checkpoint["iteration"]
            self.logger.info(f"Loading {iteration=}.")
        else:
            iteration = 0

        return iteration

    def on_iteration(self, iteration: int, encoder: nn.Module, decoder: Optional[nn.Module] = None) -> None:
        if decoder is not None and iteration % self.config.evaluation_period == 0:
            metrics = self._maybe_update_and_get_metrics(iteration, encoder, decoder)
            for key, value in metrics.items():
                if value is None:
                    return
                self.writer.add_scalar(f"test_metrics_means/{key}", value[0], iteration)  # type: ignore
                self.writer.add_scalar(f"test_metrics_stds/{key}", value[1], iteration)  # type: ignore

    def _maybe_update_and_get_metrics(self, iteration: int, encoder: nn.Module, decoder: nn.Module) -> MetricsDictionary:
        if iteration > self.last_metrics_iteration:
            time_start = time.time()
            self.metrics = get_metrics(
                self.test_dataset,
                encoder,
                decoder,
                self.shared_config,
                self.device,
                self.config.lpips_batch,
                self.config.Ts_to_evaluate,
                self.config.taus_to_evaluate,
            )
            self.logger.info(f"Calculated metrics, took {(time.time() - time_start):.2f} seconds.")
            self.last_metrics_iteration = iteration
        return self.metrics

    def save_checkpoint(
        self,
        iteration: int,
        encoding_network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        decoding_network: Optional[nn.Module] = None,
    ) -> None:
        checkpoint_path = self.subrun_directory / LATEST_CHECKPOINT_FILENAME
        checkpoint: Checkpoint = {
            "encoding_state_dict": encoding_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "decoding_state_dict": decoding_network.state_dict() if decoding_network else None,
            "iteration": iteration,
        }
        torch.save(checkpoint, checkpoint_path)

        if iteration % self.config.persistent_saving_frequency == 0:
            persistent_checkpoint_path = self.subrun_directory / PERSISTENT_CHECKPOINT_TEMPLATE.format(i=iteration)
            torch.save(checkpoint, persistent_checkpoint_path)
            if decoding_network is not None:
                self._visualize_persistent(iteration, encoding_network, decoding_network)

    def _get_previous_and_current_subrun_directories(
        self,
    ) -> Tuple[Optional[Path], Path]:
        if self.training_config.training_monitor.run_directory:
            max_dir = self._get_maximum_existing_subrun_directory_number(self.training_config.training_monitor.run_directory)
            previous_subrun_directory = self.training_config.training_monitor.run_directory / f"subrun_{max_dir:02d}" if max_dir >= 0 else None

            return (
                previous_subrun_directory,
                self.training_config.training_monitor.run_directory / f"subrun_{max_dir+1:02d}",
            )

        return None, Path("runs", datetime.now().strftime("%b%d_%H-%M-%S"), "subrun_00")

    @staticmethod
    def _get_maximum_existing_subrun_directory_number(run_directory: Path) -> int:
        max_dir = -1

        if run_directory.exists():
            for filename in os.listdir(run_directory):
                # Use regex to check if the folder name fits the pattern
                match = re.match(r"subrun_(\d{2})$", filename)
                if match:
                    # If it does, get the number
                    number = int(match.group(1))
                    # Update the maximum number found so far
                    max_dir = max(max_dir, number)

        return max_dir

    def on_reconstruction(self, image_loss: float, iteration: int) -> None:
        self.writer.add_scalar("reconstruction/ImageLoss", image_loss, iteration)  # type: ignore[no-untyped-call]

    def visualize(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time 1 X Y"],
        images: Float32[np.ndarray, "batch Time 1 X Y"],
        predictions: Float32[np.ndarray, "batch Time 1 XUpscaledCropped YUpscaledCropped"],
        iteration: int,
        encoding_network: Optional[nn.Module] = None,
        decoding_network: Optional[nn.Module] = None,
    ) -> None:
        video_event_polarity_sums, video_images, video_predictions, video_stds = self._generate_montage(fused_event_polarity_sums, images, predictions)

        videos = [video_event_polarity_sums, video_images, video_predictions]
        if video_stds is not None:
            videos.append(video_stds)

        montage_frames = np.stack(videos, axis=0)  # S B T C X Y

        montage_frames_video = einops.rearrange(montage_frames, "S B T C X Y -> 1 T C (B X) (S Y)")
        self.writer.add_video("reconstruction_visualization", montage_frames_video, iteration)  # type: ignore[no-untyped-call]

        if encoding_network is None or decoding_network is None:
            self.writer.flush()  # type: ignore[no-untyped-call]
            return

        if not self.shared_config.spatial_upscaling:
            pass
            # x_t_plot = self._generate_x_t_plot(encoding_network, decoding_network)
            # self.writer.add_image("last 5 frames", to_numpy(x_t_plot), iteration)  # type: ignore[no-untyped-call]

        self.writer.flush()  # type: ignore[no-untyped-call]

    def visualize_upsampling(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time 1 XDown YDown"],
        images: Float32[np.ndarray, "batch Time 1 X Y"],
        predictions: Float32[np.ndarray, "batch Time C X Y"],
        iteration: int,
        encoding_network: Optional[nn.Module],
        decoding_network: Optional[nn.Module],
    ) -> None:
        video_event_polarity_sums, video_images, video_predictions, video_stds = self._generate_montage(fused_event_polarity_sums, images, predictions)

        dowscaled_size = fused_event_polarity_sums.shape[-2:]
        upscaled_size = images.shape[-2:]

        video_for_downscaling = einops.rearrange(video_images, "B T C X Y -> (B T) X Y C")
        downscaled_frames = [
            skimage.transform.resize(video_frame, output_shape=dowscaled_size, order=3, anti_aliasing=True) for video_frame in video_for_downscaling
        ]
        upscaled_frames = [skimage.transform.resize(video_frame, upscaled_size, order=0) for video_frame in downscaled_frames]
        upscaled_video = np.stack(upscaled_frames, axis=0)
        upscaled_video = einops.rearrange(upscaled_video, "(B T) X Y C -> B T C X Y", B=video_event_polarity_sums.shape[0])

        video_eps = einops.rearrange(torch.tensor(video_event_polarity_sums), "batch Time C X Y -> (batch Time) C X Y")
        video_eps = resize(video_eps, upscaled_size, interpolation=InterpolationMode.NEAREST).numpy()
        video_eps = einops.rearrange(video_eps, "(batch Time) C X Y -> batch Time C X Y", batch=video_images.shape[0])

        videos = [upscaled_video, video_eps, video_images, video_predictions]
        if video_stds is not None:
            videos.append(video_stds)

        montage_frames = np.stack(videos, axis=0)  # S B T C X Y

        montage_frames_video = einops.rearrange(montage_frames, "S B T C X Y -> 1 T C (B X) (S Y)")
        self.writer.add_video("reconstruction_visualization", montage_frames_video, iteration)  # type: ignore[no-untyped-call]

        if encoding_network is None or decoding_network is None:
            self.writer.flush()  # type: ignore[no-untyped-call]
            return

        if not self.shared_config.spatial_upscaling:
            pass
            # x_t_plot = self._generate_x_t_plot(encoding_network, decoding_network)
            # self.writer.add_image("last 5 frames", to_numpy(x_t_plot), iteration)  # type: ignore[no-untyped-call]

        self.writer.flush()  # type: ignore[no-untyped-call]

    def _generate_montage(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time C X Y"],
        images: Float32[np.ndarray, "batch Time 1 X Y"],
        predictions: Float32[np.ndarray, "batch Time C X Y"],
    ) -> Tuple[
        Float32[np.ndarray, "batch Time 3 X Y"],
        Float32[np.ndarray, "batch Time 3 X Y"],
        Float32[np.ndarray, "batch Time 3 X Y"],
        Optional[Float32[np.ndarray, "batch Time 3 X Y"]],
    ]:
        colored_event_polarity_sums = img_to_colormap(fused_event_polarity_sums.sum(axis=2), create_red_blue_cmap(501))
        colored_event_polarity_sums = einops.rearrange(colored_event_polarity_sums, "B T X Y C -> B T C X Y")

        images = scale_to_quantiles(images, [1, 2, 3, 4], 0.005, 0.995)
        images = einops.repeat(images, "batch Time 1 X Y -> batch Time C X Y", C=3)

        means = predictions[:, :, 0:1]
        means = scale_to_quantiles(means, [1, 2, 3, 4], 0.005, 0.995)
        means = einops.repeat(means, "batch Time 1 X Y -> batch Time C X Y", C=3)

        if predictions.shape[2] > 1:
            stds = np.exp(predictions[:, :, 1:2])
            stds = scale_to_quantiles(stds, [1, 2, 3, 4], 0.005, 0.995)
            stds = einops.repeat(stds, "batch Time 1 X Y -> batch Time C X Y", C=3)
        else:
            stds = None

        return colored_event_polarity_sums, images, means, stds

    @torch.no_grad()
    def _generate_x_t_plot(
        self,
        encoding_network: nn.Module,
        decoding_network: nn.Module,
    ) -> Float32[torch.Tensor, "X T"]:
        n_taus = 5

        eps, means, stds, counts, down_eps, down_means, down_stds, down_counts, preprocessed_image, transform = network_test_data_to_device(
            self.sample_batch, self.device, self.shared_config.use_mean, self.shared_config.use_std, self.shared_config.use_count
        )

        cs_list = []
        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            c_t = encoding_network(
                eps[:, t],
                means[:, t] if self.shared_config.use_mean else None,
                stds[:, t] if self.shared_config.use_std else None,
                counts[:, t] if self.shared_config.use_count else None,
            )
            # Save last 8 encodings
            if t >= self.shared_config.sequence_length - 8:
                cs_list.append(c_t.clone())

        # Unfold c
        batch_size, X_size = eps.shape[0], eps.shape[-2]
        cs = torch.stack(cs_list, dim=0)  # T B C X Y
        if self.shared_config.spatial_unfolding:
            cs = einops.rearrange(cs, "T B C X Y -> (T B) C X Y")
            cs = torch.nn.functional.unfold(cs, kernel_size=(3, 3), padding=(1, 1), stride=1)
            cs = einops.rearrange(cs, "(T B) C (X Y) -> T B C X Y", T=len(cs_list), X=X_size)
        # Prepare for linear layer
        cs = einops.rearrange(cs, "T B C X Y -> T B X Y C")

        c_cols = [x[:, :, 0, :] for x in cs]  # type: ignore

        rs = []

        for i in range(1, 6):
            c, c_next = c_cols[i], c_cols[i + 1]  # B X C
            if self.shared_config.temporal_unfolding:
                c = torch.concat([c_cols[i - 1], c_cols[i], c_cols[i + 1]], dim=-1)
                c_next = torch.concat([c_cols[i], c_cols[i + 1], c_cols[i + 2]], dim=-1)

            for tau in torch.arange(0, 1 - 1e-5, 1 / n_taus):
                tau = einops.repeat(torch.tensor([tau]).to(c), "1 -> B X 1", B=c.shape[0], X=c.shape[1])
                r_t = decoding_network(torch.concat([c, tau], dim=-1))
                if self.shared_config.temporal_interpolation:
                    r_tnext = decoding_network(torch.concat([c_next, tau - 1], dim=-1))
                    r_t = r_t * (1 - tau) + r_tnext * (tau)
                rs.append(r_t)

        # Reconstructions is now a temporally ordered list of 8*n_taus (batch X 1) tensors
        stacked_reconstructions = torch.stack(rs, dim=0)  # 8*n_taus, batch, X 1
        images = einops.rearrange(stacked_reconstructions, "T batch X 1 -> batch X T")
        color_images = einops.repeat(images, "batch X T -> batch 3 X T")
        red_strip = einops.repeat(torch.tensor([1, 0, 0]).to(c), "C -> C X 1", X=color_images.shape[2])
        images_to_concat = []
        for i, color_image in enumerate(color_images):
            images_to_concat.append(color_image)
            if i < len(color_images):
                images_to_concat.append(red_strip)

        return torch.concat(images_to_concat, dim=2)

    @torch.no_grad()
    def _visualize_persistent(self, iteration: int, encoder: nn.Module, decoder: nn.Module) -> None:
        self.logger.info("Calculating metrics...")
        metrics = self._maybe_update_and_get_metrics(iteration, encoder, decoder)
        self.logger.info(f"Calculated metrics: {metrics}")

        I = len(self.config.test_samples_to_visualize)
        for idx, i in enumerate(self.config.test_samples_to_visualize):
            self.logger.info(f"{idx}/{I} - getting data...")
            batch = collate_test_items([self.test_dataset[i]])
            name = self.test_dataset.directory_list[i].name
            scale = self.test_dataset.scales[i]

            recon, gt, gt_down, eps = get_reconstructions_and_gt(
                batch, encoder, decoder, self.shared_config, self.device, scale, self.config.Ts_to_visualize, self.config.taus_to_visualize
            )

            self.logger.info(f"{idx}/{I} - generating video...")
            ani = get_evaluation_video(recon, gt, gt_down, eps, range(self.config.Ts_to_visualize * self.config.taus_to_visualize), scale)

            self.logger.info(f"{idx}/{I} - saving video...")
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
                ani.save(temp_file.name, writer="ffmpeg", fps=10)
                vid, _, _ = read_video(temp_file.name)

            vid = einops.rearrange(vid, "T H W C -> 1 T C H W")
            self.writer.add_video(f"video_{name}", vid, iteration)  # type: ignore

            self.logger.info(f"{idx}/{I} - generating figure...")
            figure = get_evaluation_image(metrics, recon, gt, gt_down, eps, scale)
            plt.close()
            self.logger.info(f"{idx}/{I} - saving figure...")
            self.writer.add_figure(f"figure_{name}", figure, iteration)  # type: ignore
            self.writer.flush()  # type: ignore
