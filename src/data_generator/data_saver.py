import logging
import shutil
from pathlib import Path
from typing import Dict

import h5py

from on_the_fly.trainers.utils.datatypes import GrayVideo, GrayVideoTorch, Image
from on_the_fly.trainers.utils.discretized_events import DiscretizedEvents

from .configuration import DataSaverConfiguration


class DataSaver:
    config: DataSaverConfiguration
    logger: logging.Logger

    def __init__(self, config: DataSaverConfiguration) -> None:
        self.config = config
        self.logger = logging.getLogger("DataSaver")

    def run(  # pylint: disable=too-many-arguments
        self,
        image_path: Path,
        image: Image,
        video: GrayVideo,
        discretized_events_dict: Dict[float, DiscretizedEvents],
        synchronized_logarithmic_video: GrayVideoTorch,
    ) -> None:
        self.logger.info("Saving data...")

        output_dir = Path(self.config.output_dir) / image_path.stem

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            for threshold, discretized_events in discretized_events_dict.items():
                with h5py.File(
                    output_dir / f"discretized_events_{threshold}.h5", "w"
                ) as file:
                    discretized_events.save_to_file(file, self.config.h5_compression)

            with h5py.File(output_dir / "input.h5", "w") as file:
                file.create_dataset(
                    "/input_image",
                    data=image,
                    compression="gzip",
                    compression_opts=self.config.h5_compression,
                )
                file.create_dataset(
                    "/generated_video",
                    data=video,
                    compression="gzip",
                    compression_opts=self.config.h5_compression,
                )

            with h5py.File(output_dir / "ground_truth.h5", "w") as file:
                file.create_dataset(
                    "/synchronized_video",
                    data=synchronized_logarithmic_video.cpu(),
                    compression="gzip",
                    compression_opts=self.config.h5_compression,
                )
        except Exception:  # pylint: disable=broad-exception-caught
            logging.error("Exception in data saving - deleting files!")
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    def output_exists(self, image_path: Path) -> bool:
        output_dir = Path(self.config.output_dir) / image_path.stem
        return output_dir.exists()
