import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float32
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dynamic_fusion.utils.datatypes import Batch, Checkpoint
from dynamic_fusion.utils.image import scale_to_quantiles_numpy
from dynamic_fusion.utils.network import network_data_to_device, to_numpy

from .configuration import SharedConfiguration, TrainerConfiguration

LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.pt"


class TrainingMonitor:
    """This class handles checkpoints and logging"""

    training_config: TrainerConfiguration
    writer: SummaryWriter
    sample_batch: Batch
    device: torch.device
    shared_config: SharedConfiguration
    logger: logging.Logger
    subrun_directory: Path

    def __init__(self, training_config: TrainerConfiguration) -> None:
        self.training_config = training_config
        self.config = training_config.training_monitor
        self.shared_config = training_config.shared
        self.logger = logging.getLogger("TrainingMonitor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(
        self,
        data_loader: DataLoader,  # type: ignore
        reconstruction_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        decoding_network: Optional[nn.Module],
    ) -> int:
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
        self.sample_batch = next(iter(data_loader))
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

        checkpoint: Checkpoint = torch.load(checkpoint_path)
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

    def save_checkpoint(
        self,
        encoding_network: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        decoding_network: Optional[nn.Module] = None,
        iteration: Optional[int] = None,
    ) -> None:
        checkpoint_path = self.subrun_directory / LATEST_CHECKPOINT_FILENAME
        checkpoint: Checkpoint = {
            "encoding_state_dict": encoding_network.state_dict() if encoding_network else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "decoding_state_dict": decoding_network.state_dict() if decoding_network else None,
            "iteration": iteration,
        }
        torch.save(checkpoint, checkpoint_path)

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
        predictions: Float32[np.ndarray, "batch Time 1 X Y"],
        iteration: int,
        encoding_network: Optional[nn.Module] = None,
        decoding_network: Optional[nn.Module] = None,
    ) -> None:
        video_event_polarity_sums, video_images, video_predictions = self._generate_montage(fused_event_polarity_sums, images, predictions)

        montage_frames = np.stack(
            (video_event_polarity_sums, video_images, video_predictions),
            axis=0,
        )  # S B T C X Y
        montage_frames_video = einops.rearrange(montage_frames, "S B T C X Y -> 1 T C (B X) (S Y)")
        self.writer.add_video("reconstruction_visualization", montage_frames_video, iteration)  # type: ignore[no-untyped-call]

        if encoding_network is None or decoding_network is None:
            self.writer.flush()  # type: ignore[no-untyped-call]
            return

        x_t_plot = self._generate_x_t_plot(encoding_network, decoding_network)

        self.writer.add_image("last 5 frames", to_numpy(x_t_plot), iteration)  # type: ignore[no-untyped-call]
        self.writer.flush()  # type: ignore[no-untyped-call]

    def _generate_montage(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time 1 X Y"],
        images: Float32[np.ndarray, "batch Time 1 X Y"],
        predictions: Float32[np.ndarray, "batch Time 1 X Y"],
    ) -> Tuple[
        Float32[np.ndarray, "batch Time 3 X Y"],
        Float32[np.ndarray, "batch Time 3 X Y"],
        Float32[np.ndarray, "batch Time 3 X Y"],
    ]:
        colored_event_polarity_sums = self.img_to_colormap(fused_event_polarity_sums[:, :, 0, :, :], self.create_red_blue_cmap(501))
        colored_event_polarity_sums = einops.rearrange(colored_event_polarity_sums, "B T X Y C -> B T C X Y")
        images = scale_to_quantiles_numpy(images, [1, 2, 3, 4], 0.005, 0.995)
        predictions = scale_to_quantiles_numpy(predictions, [1, 2, 3, 4], 0.005, 0.995)

        images = einops.repeat(images, "batch Time 1 X Y -> batch Time C X Y", C=3)
        predictions = einops.repeat(predictions, "batch Time 1 X Y -> batch Time C X Y", C=3)

        return colored_event_polarity_sums, images, predictions

    @torch.no_grad()
    def _generate_x_t_plot(
        self,
        encoding_network: nn.Module,
        decoding_network: nn.Module,
    ) -> Float32[torch.Tensor, "X T"]:
        n_taus = 5

        event_polarity_sums, timestamp_means, timestamp_stds, event_counts, video, preprocessed_images, transforms, crops = network_data_to_device(
            self.sample_batch, self.device, self.shared_config.use_mean, self.shared_config.use_std, self.shared_config.use_count
        )

        cs_list = []
        for t in range(self.shared_config.sequence_length):  # pylint: disable=C0103
            c_t = encoding_network(
                event_polarity_sums[:, t],
                timestamp_means[:, t] if self.shared_config.use_mean else None,
                timestamp_stds[:, t] if self.shared_config.use_std else None,
                event_counts[:, t] if self.shared_config.use_count else None,
            )
            # Save last 8 encodings
            if t >= self.shared_config.sequence_length - 8:
                cs_list.append(c_t.clone())

        # Unfold c
        batch_size, X_size = event_polarity_sums.shape[0], event_polarity_sums.shape[-2]
        cs = torch.stack(cs_list, dim=0)  # T B C X Y
        if self.shared_config.spatial_unfolding:
            cs = einops.rearrange(cs, "T B C X Y -> (T B C) X Y")
            cs = torch.nn.functional.unfold(cs, kernel_size=(3, 3), padding=(1, 1), stride=1)
            cs = einops.rearrange(cs, "(T B C) (X Y) -> T B C X Y", T=len(cs_list), B=batch_size, X=X_size)
        # Prepare for linear layer
        cs = einops.rearrange(cs, "T B C X Y -> T B X Y C")

        c_cols = [x[:, :, 0, :] for x in cs]  # type: ignore

        rs = []

        for i in range(1, 6):
            c, c_next = c_cols[i], c_cols[i + 1]  # B X C
            if self.shared_config.temporal_unfolding:
                c = torch.concat([c_cols[i - 1], c_cols[i], c_cols[i + 1]], dim=-1)  # type: ignore
                c_next = torch.concat([c_cols[i], c_cols[i + 1], c_cols[i + 2]], dim=-1)  # type: ignore

            for tau in torch.arange(0, 1 - 1e-5, 1 / n_taus):
                tau = einops.repeat(torch.tensor([tau]).to(c), "T -> T X 1", X = c.shape[-2])
                r_t = decoding_network(torch.concat([c, tau], dim=-1))  # type: ignore
                if self.shared_config.temporal_interpolation:
                    r_tnext = decoding_network(torch.concat([c_next, tau - 1], dim=-1))  # type: ignore
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

    # TODO: this can be reused and should be rewritten
    @staticmethod
    def create_red_blue_cmap(N) -> np.ndarray:  # type: ignore
        assert N % 2 == 1
        cm = np.zeros([N, 3])
        cm[: N // 2, 0] = np.sqrt(np.linspace(N / 2, 1, N // 2) * 2 / N)
        cm[N // 2 :, 2] = np.sqrt(np.linspace(1, N / 2, N - N // 2) * 2 / N)
        cm[N // 2, :] = 0.5
        return cm

    @staticmethod
    def img_to_colormap(img, cmap, clims=None) -> np.ndarray:  # type: ignore
        if clims is None:
            clims = np.array([-1, 1]) * np.max(np.abs(img))
        N, C = cmap.shape
        # assert N % 2 == 1
        grid_x = np.linspace(clims[0], clims[1], N)
        img_colored = np.zeros(img.shape + (C,))
        for i in range(C):  # 3 color channels
            img_colored[..., i] = np.interp(img, grid_x, cmap[:, i])
        return img_colored
