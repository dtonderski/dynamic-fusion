import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float32
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dynamic_fusion.utils.datatypes import Checkpoint
from dynamic_fusion.utils.image import scale_to_quantiles_numpy

from .configuration import (
    SharedConfiguration,
    TrainerConfiguration,
)
from .utils.datatypes import Batch

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
        decoding_network: nn.Module,
    ) -> int:
        previous_subrun_directory, self.subrun_directory = (
            self._get_previous_and_current_subrun_directories()
        )
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
        reconstruction_network: nn.Module,
        reconstruction_optimizer: torch.optim.Optimizer,
        decoding_network: nn.Module,
    ) -> int:
        checkpoint_path = previous_subrun_directory / LATEST_CHECKPOINT_FILENAME
        if not checkpoint_path.exists():
            raise ValueError(f"Subrun found, but no {LATEST_CHECKPOINT_FILENAME}!")

        checkpoint: Checkpoint = torch.load(checkpoint_path)
        if checkpoint["reconstruction_state_dict"]:
            self.logger.info("Loading reconstruction_state_dict.")
            reconstruction_network.load_state_dict(
                checkpoint["reconstruction_state_dict"]
            )
            reconstruction_network.to(self.device)

        if checkpoint["decoding_state_dict"]:
            self.logger.info("Loading decoding_state_dict.")
            decoding_network.load_state_dict(
                checkpoint["decoding_state_dict"]
            )
            decoding_network.to(self.device)

        if checkpoint["optimizer_state_dict"]:
            self.logger.info("Loading optimizer_state_dict.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if checkpoint["iteration"]:
            iteration = checkpoint["iteration"]
            self.logger.info(f"Loading {iteration=}.")
        else:
            iteration = 0

        return iteration

    def save_checkpoint(
        self,
        reconstruction_network: Optional[nn.Module] = None,
        reconstruction_optimizer: Optional[torch.optim.Optimizer] = None,
        decoding_network: Optional[nn.Module] = None,
        iteration: Optional[int] = None,
    ) -> None:
        checkpoint_path = self.subrun_directory / LATEST_CHECKPOINT_FILENAME
        checkpoint: Checkpoint = {
            "reconstruction_state_dict": (
                reconstruction_network.state_dict()
                if reconstruction_network
                else None
            ),
            "reconstruction_optimizer_state_dict": (
                reconstruction_optimizer.state_dict()
                if reconstruction_optimizer
                else None
            ),
            "decoding_state_dict": (
                decoding_network.state_dict()
                if decoding_network
                else None
            ),
            "iteration": iteration,
        }
        torch.save(checkpoint, checkpoint_path)

    def _get_previous_and_current_subrun_directories(
        self,
    ) -> Tuple[Optional[Path], Path]:
        if self.training_config.training_monitor.run_directory:
            max_dir = self._get_maximum_existing_subrun_directory_number(
                self.training_config.training_monitor.run_directory
            )
            previous_subrun_directory = (
                self.training_config.training_monitor.run_directory
                / f"subrun_{max_dir:02d}"
                if max_dir >= 0
                else None
            )

            return (
                previous_subrun_directory,
                self.training_config.training_monitor.run_directory
                / f"subrun_{max_dir+1:02d}",
            )

        return None, Path(
            "runs", datetime.now().strftime("%b%d_%H-%M-%S"), "subrun_00"
        )

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
        self.writer.add_scalar(
            "reconstruction/ImageLoss", image_loss, iteration
        )  # type: ignore[no-untyped-call]

    def visualize(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time 1 X Y"],
        images: Float32[np.ndarray, "batch Time 1 X Y"],
        predictions: Float32[np.ndarray, "batch Time 1 X Y"],
        iteration: int,
        video_only: bool,
    ) -> None:
        video_event_polarity_sums, video_images, video_predictions = (
            self._generate_montage(fused_event_polarity_sums, images, predictions)
        )

        montage_frames = np.stack(
            (video_event_polarity_sums, video_images, video_predictions),
            axis=0,
        )  # S B T C X Y
        montage_frames_video = einops.rearrange(
            montage_frames, "S B T C X Y -> 1 T C (B X) (S Y)"
        )
        self.writer.add_video(
            "reconstruction_visualization", montage_frames_video, iteration
        )  # type: ignore[no-untyped-call]

        if video_only:
            self.writer.flush()  # type: ignore[no-untyped-call]
            return

        self._save_mask_and_event_visualization(
            video_event_polarity_sums, video_images, iteration
        )
        self.writer.flush()  # type: ignore[no-untyped-call]

    def _save_mask_and_event_visualization(
        self,
        fused_event_polarity_sums: Float32[np.ndarray, "batch Time 3 X Y"],
        images: Float32[np.ndarray, "batch Time 3 X Y"],
        iteration: int,
    ) -> None:
        visualization = np.concatenate([images, fused_event_polarity_sums], axis=3)
        visualization = einops.rearrange(visualization, "B T C X Y -> 1 T C X (B Y)")
        self.writer.add_video(
            "mask_and_event_visualization", visualization, iteration
        )  # type: ignore[no-untyped-call]

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
        colored_event_polarity_sums = self._img_to_colormap(
            fused_event_polarity_sums[:, :, 0, :, :], self._create_red_blue_cmap(501)
        )
        colored_event_polarity_sums = einops.rearrange(
            colored_event_polarity_sums, "B T X Y C -> B T C X Y"
        )
        images = scale_to_quantiles_numpy(images, [1, 2, 3, 4], 0.005, 0.995)
        predictions = scale_to_quantiles_numpy(
            predictions, [1, 2, 3, 4], 0.005, 0.995
        )

        images = einops.repeat(images, "batch Time 1 X Y -> batch Time C X Y", C=3)
        predictions = einops.repeat(
            predictions, "batch Time 1 X Y -> batch Time C X Y", C=3
        )

        return colored_event_polarity_sums, images, predictions

    # TODO: this can be reused and should be rewritten
    def _create_red_blue_cmap(self, N) -> np.ndarray:  # type: ignore
        assert N % 2 == 1
        cm = np.zeros([N, 3])
        cm[: N // 2, 0] = np.sqrt(np.linspace(N / 2, 1, N // 2) * 2 / N)
        cm[N // 2 :, 2] = np.sqrt(np.linspace(1, N / 2, N - N // 2) * 2 / N)
        cm[N // 2, :] = 0.5
        return cm

    def _img_to_colormap(self, img, cmap, clims=None) -> np.ndarray:  # type: ignore
        if clims is None:
            clims = np.array([-1, 1]) * np.max(np.abs(img))
        N, C = cmap.shape
        # assert N % 2 == 1
        grid_x = np.linspace(clims[0], clims[1], N)
        img_colored = np.zeros(img.shape + (C,))
        for i in range(C):  # 3 color channels
            img_colored[..., i] = np.interp(img, grid_x, cmap[:, i])
        return img_colored
