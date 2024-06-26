import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple

import h5py

from dynamic_fusion.utils.datatypes import Events, GrayImageFloat, GrayVideoFloat, Image
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.transform import TransformDefinition

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
        video: GrayVideoFloat,
        downscaled_video: GrayVideoFloat,
        preprocessed_image: GrayImageFloat,
        transform_definition: TransformDefinition,
        event_dict: Dict[float, Events],
        downscaled_event_dict: Dict[float, Events],
        discretized_events_dict: Dict[float, DiscretizedEvents],
        downscaled_discretized_events_dict: Dict[float, DiscretizedEvents],
        exponentiation_multiplier: float,
        illuminance_range: Tuple[float, float],
        subbins: int
    ) -> None:
        self.logger.info("Saving data...")
        output_dir = self.config.output_dir / image_path.stem

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            for threshold, discretized_events in discretized_events_dict.items():
                with h5py.File(output_dir / f"discretized_events_{threshold}_{subbins}.h5", "w") as file:
                    discretized_events.save_to_file(file, self.config.h5_compression)

            for threshold, discretized_events in downscaled_discretized_events_dict.items():
                with h5py.File(output_dir / f"downscaled_discretized_events_{threshold}_{subbins}.h5", "w") as file:
                    discretized_events.save_to_file(file, self.config.h5_compression)

            with h5py.File(output_dir / "input.h5", "w") as file:
                file.create_dataset("/input_image", data=image, compression="gzip", compression_opts=self.config.h5_compression)

                if self.config.save_video:
                    file.create_dataset("/generated_video", data=video, compression="gzip", compression_opts=self.config.h5_compression)

                file.create_dataset("/preprocessed_image", data=preprocessed_image)
                file.create_dataset("/exponentiation_multiplier", data=exponentiation_multiplier)
                file.create_dataset("/illuminance_range", data=illuminance_range)

                transform_definition.save_to_file(file)

            with h5py.File(output_dir / "downscaled_input.h5", "w") as file:
                if self.config.save_video:
                    file.create_dataset("/generated_video", data=downscaled_video, compression="gzip", compression_opts=self.config.h5_compression)

            # Read using data = pd.read_hdf("events.h5", "threshold..."")
            if self.config.save_events:
                if event_dict is not None:
                    for threshold, event_df in event_dict.items():
                        event_df.to_hdf(output_dir / "events.h5", f"threshold{threshold}", "a", complevel=self.config.h5_compression, complib="zlib")
                for threshold, downscaled_event_df in downscaled_event_dict.items():
                    downscaled_event_df.to_hdf(
                        output_dir / "downscaled_events.h5", f"threshold{threshold}", "a", complevel=self.config.h5_compression, complib="zlib"
                    )

        except Exception:  # pylint: disable=broad-exception-caught
            logging.error("Exception in data saving - deleting files!")
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    def output_exists(self, image_path: Path) -> bool:
        output_dir = Path(self.config.output_dir) / image_path.stem
        return output_dir.exists()
