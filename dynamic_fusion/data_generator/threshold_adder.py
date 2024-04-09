import logging
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

from dynamic_fusion.data_generator.video_generator import VideoGenerator
from dynamic_fusion.utils.datatypes import GrayImageFloat
from dynamic_fusion.utils.network import to_numpy
from dynamic_fusion.utils.transform import TransformDefinition
from dynamic_fusion.utils.video import get_video, normalize

from .configuration import DataGeneratorConfiguration
from .event_discretizer import EventDiscretizer
from .event_generator import EventGenerator


class ThresholdAdder:
    """Class that adds one or more thresholds to an existing dataset"""

    logger: logging.Logger

    original_config: DataGeneratorConfiguration
    event_generator: EventGenerator
    event_discretizer: EventDiscretizer

    def __init__(self, thresholds: List[float], original_config: DataGeneratorConfiguration) -> None:
        self.logger = logging.getLogger("DataGenerator")

        self.original_config = original_config
        self.video_generator = VideoGenerator(original_config.video_generator, original_config.shared)
        original_config.event_generator.thresholds = thresholds
        self.event_generator = EventGenerator(original_config.event_generator, original_config.shared)

    def run(self) -> None:
        existing_output_dirs = self.original_config.data_saver.output_dir.glob("*/**")

        for output_dir in tqdm(existing_output_dirs):
            if (output_dir / f"events_{threshold}.h5").exists() and (output_dir / f"downscaled_events_{threshold}.h5").exists():
                continue

            with h5py.File(output_dir / "input.h5", "r") as file:
                preprocessed_image: GrayImageFloat = np.array(file["preprocessed_image"])
                transform_definition = TransformDefinition.load_from_file(file)
                exponentiation_multiplier = float(np.array(file["exponentiation_multiplier"]))
                illuminance_range = tuple(file["illuminance_range"])

            exponentiated_image = np.exp(preprocessed_image * exponentiation_multiplier)
            exponentiated_image = normalize(exponentiated_image)

            downscaling_factor = self.original_config.shared.downscaling_factor
            downscaled_resolution = tuple(int(x / downscaling_factor) for x in preprocessed_image.shape)
            self.logger.info(f"{downscaled_resolution=}")

            video_frame_times = np.linspace(0, 1, self.original_config.shared.number_of_images_to_generate_per_input)

            downscaled_video = get_video(
                exponentiated_image,
                transform_definition,
                video_frame_times,
                fill_mode=self.original_config.video_generator.fill_mode,
                downscale=True,
                try_center_crop=False,
            )
            downscaled_video = to_numpy(downscaled_video)

            video = get_video(
                exponentiated_image,
                transform_definition,
                video_frame_times,
                fill_mode=self.original_config.video_generator.fill_mode,
                downscale=False,
                try_center_crop=True,
            )
            video = to_numpy(video)

            unscaled_event_dict, illuminance_range = self.event_generator.run(video, illuminance_range=illuminance_range)

            downscaled_event_dict, _ = self.event_generator.run(downscaled_video, illuminance_range=illuminance_range)

            for threshold, unscaled_event_df in unscaled_event_dict.items():
                unscaled_event_df.to_hdf(
                    output_dir / f"events_{threshold}.h5", f"threshold{threshold}", "a", complevel=self.original_config.data_saver.h5_compression, complib="zlib"
                )

            for threshold, downscaled_event_df in downscaled_event_dict.items():
                downscaled_event_df.to_hdf(
                    output_dir / f"downscaled_events_{threshold}.h5", f"threshold{threshold}", "a", complevel=self.original_config.data_saver.h5_compression, complib="zlib"
                )
