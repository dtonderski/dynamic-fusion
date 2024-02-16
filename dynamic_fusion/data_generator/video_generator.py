import logging
from typing import Literal, Optional, Tuple

import numpy as np
from jaxtyping import Float
from numpy.random import randint, uniform
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import GrayImage, GrayVideoFloat
from dynamic_fusion.utils.network import to_numpy
from dynamic_fusion.utils.transform import TransformDefinition
from dynamic_fusion.utils.video import get_video

from .configuration import SharedConfiguration, VideoGeneratorConfiguration


class VideoGenerator:
    config: VideoGeneratorConfiguration
    shared_config: SharedConfiguration
    logger: logging.Logger

    def __init__(
        self,
        config: VideoGeneratorConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        self.logger = logging.getLogger("VideoGenerator")

    def run(self, image: GrayImage, transform_definition: Optional[TransformDefinition] = None) -> Tuple[GrayVideoFloat, TransformDefinition]:
        self.logger.info("Generating video...")

        if transform_definition is None:
            transform_definition = self._define_transforms()

        video_frame_times = np.linspace(0, 1, self.shared_config.number_of_images_to_generate_per_input)

        video = get_video(
            image,
            transform_definition,
            video_frame_times,
            self.shared_config.target_image_size,
            fill_mode=self.config.fill_mode,
            interpolation=self.config.interpolation,
        )
        return to_numpy(video), transform_definition

    def _define_transforms(self) -> TransformDefinition:
        def _generate_interpolation_type() -> Literal["linear", "cubic"]:
            return "linear" if uniform() > 0.5 else "cubic"

        shift_knots, rotation_knots, scale_knots = self._generate_knots()
        shift_interpolation, rotation_interpolation, scale_interpolation = (_generate_interpolation_type() for _ in range(3))
        return TransformDefinition(
            shift_knots,
            rotation_knots,
            scale_knots,
            shift_interpolation,
            rotation_interpolation,
            scale_interpolation,
        )

    def _generate_knots(self) -> Tuple[
        Float[np.ndarray, "NShiftKnots 2"],
        Float[np.ndarray, "NRotKnots 1"],
        Float[np.ndarray, "NScaleKnots 2"],
    ]:
        number_of_shift_knots = randint(low=2, high=self.config.max_number_of_shift_knots, dtype=np.int32)

        shift_knot_values = uniform(size=(number_of_shift_knots, 2)) * self.config.max_shift_knot_multiplier_value

        number_of_rotation_knots = randint(
            low=2,
            high=self.config.max_number_of_rotation_knots,
            dtype=np.int32,
        )
        rotation_knot_values = uniform(size=(number_of_rotation_knots, 1)) * self.config.max_rotation_knot_value * uniform()

        number_of_scale_knots = randint(low=2, high=self.config.max_number_of_scale_knots, dtype=np.int32)
        scale_knot_values = 1.0 + (uniform(low=-0.5, high=0.5, size=(number_of_scale_knots, 2)) * self.config.max_scale_knot_value * uniform())

        return shift_knot_values, rotation_knot_values, scale_knot_values
