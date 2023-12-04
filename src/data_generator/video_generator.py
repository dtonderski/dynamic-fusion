import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from numpy.random import randint, uniform
from scipy.interpolate import interp1d  # pyright: ignore
from scipy.ndimage import affine_transform  # pyright: ignore
from torchvision.transforms.functional import affine
from tqdm import tqdm

from on_the_fly.trainers.utils.datatypes import GrayImage, GrayVideo

from .configuration import SharedConfiguration, VideoGeneratorConfiguration
from .utils.video import normalize


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

    def run(self, image: GrayImage, progress_bar: Optional[tqdm] = None) -> GrayVideo:
        if progress_bar:
            progress_bar.set_postfix_str("Generating video")
        else:
            self.logger.info("Generating video...")
        shifts, rotations, scales = self._generate_transforms()
        if self.config.use_pytorch:
            angles, translates, scales = self._transforms_to_torch(
                shifts, rotations, scales
            )
            video = self._generate_video_torch(image, angles, translates, scales)
        else:
            transformation_matrices = self._transforms_to_matrices(
                image, shifts, rotations, scales
            )
            video = self._generate_video_scipy(image, transformation_matrices)
        video = normalize(video)
        video = self._crop_video(video)
        return video

    def _generate_transforms(
        self,
    ) -> Tuple[
        Float[np.ndarray, "T 2"],
        Float[np.ndarray, "T 1"],
        Float[np.ndarray, "T 2"],
    ]:
        r"""Generates shifts, rotations, and scales.

        Args:
            image (GrayImage): The image to generate transforms for. Needed
            because we need the image shape.

        Returns:
            Tuple[Shifts, Rotations, Scales]: generated transforms for each
            timestep.
        """
        number_of_shift_knots = randint(
            low=2, high=self.config.max_number_of_shift_knots, dtype=np.int32
        )
        max_shift_knot_value = (
            np.array(self.config.max_shift_knot_multiplier_value)
            * self.shared_config.target_image_size
        )
        shift_knot_values = uniform(size=(number_of_shift_knots, 2)) * np.reshape(
            max_shift_knot_value, (1, 2)
        )

        number_of_rotation_knots = randint(
            low=2,
            high=self.config.max_number_of_rotation_knots,
            dtype=np.int32,
        )
        rotation_knot_values = (
            uniform(size=(number_of_rotation_knots, 1))
            * self.config.max_rotation_knot_value
            * uniform()
        )

        number_of_scale_knots = randint(
            low=2, high=self.config.max_number_of_scale_knots, dtype=np.int32
        )
        scale_knot_values = 1.0 + (
            uniform(low=-0.5, high=0.5, size=(number_of_scale_knots, 2))
            * self.config.max_scale_knot_value
            * uniform()
        )

        shifts = self._upsample_knot_values(shift_knot_values)
        shifts -= shifts[0:1, ...]

        rotations = self._upsample_knot_values(rotation_knot_values)
        rotations -= rotations[0:1, ...]

        scales = self._upsample_knot_values(scale_knot_values)
        scales = scales - scales[0:1, ...] + 1.0

        return shifts, rotations, scales

    def _upsample_knot_values(
        self,
        knot_values: Float[np.ndarray, "NKnots Values"],
        interpolation_type: Literal["random", "linear", "cubic"] = "random",
    ) -> Float[np.ndarray, "T Values"]:
        r"""This function is used to upsample knot transformation values by
            using interpolation so that they can be used to generate output
            images.

        Args:
            knot_values (Float[np.ndarray, "NKnots Values"]):
                transformation values of the knots.
            interpolation_type (Literal, optional): inteprolation type to
                use. If "random", select randomly between "linear" and
                "cubic". Defaults to "random".

        Returns:
            Float[np.ndarray, "T Values"]: upsampled transformation values
        """

        if interpolation_type == "random":
            interpolation_type = "linear" if uniform() > 0.5 else "cubic"

        number_of_knots = knot_values.shape[0]
        if number_of_knots <= 3:
            interpolation_type = "linear"

        x_knots = np.linspace(0, 1, number_of_knots)
        interpolation = interp1d(
            x_knots, knot_values, kind=interpolation_type, axis=0
        )

        x_for_interpolation = np.linspace(
            0, 1, self.shared_config.number_of_images_to_generate_per_input
        )
        return interpolation(x_for_interpolation)

    def _transforms_to_torch(
        self,
        shifts: Float[np.ndarray, "T 2"],
        rotations: Float[np.ndarray, "T 1"],
        scales: Float[np.ndarray, "T 2"],
    ) -> Tuple[List[float], List[List[int]], List[float]]:
        angles = [float(rotation) * 180 / np.pi for rotation in rotations]
        translates = [[int(x) for x in shift] for shift in shifts]
        torch_scales = [scale[0] for scale in scales]
        return angles, translates, torch_scales

    def _transforms_to_matrices(  # pylint: disable=R0913,R0914
        self,
        image: GrayImage,
        shifts: Float[np.ndarray, "T 2"],
        rotations: Float[np.ndarray, "T 1"],
        scales: Float[np.ndarray, "T 2"],
        rotate_around_center: bool = True,
    ) -> Float[np.ndarray, "T 3 3"]:
        x_size, y_size = image.shape
        centering_matrix = np.array(
            [[1, 0, x_size / 2.0], [0, 1, y_size / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )
        inverse_centering_matrix = np.array(
            [[1, 0, -x_size / 2.0], [0, 1, -y_size / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )

        transformation_matrices = np.zeros(
            (self.shared_config.number_of_images_to_generate_per_input, 3, 3),
            dtype=np.float32,
        )

        for step in range(self.shared_config.number_of_images_to_generate_per_input):
            theta = rotations[step, 0]
            scale = scales[step, :]
            shift = shifts[step, :]

            rotation_matrix = np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            shift_matrix = np.array(
                [[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]],
                dtype=np.float32,
            )

            scale_matrix = np.array(
                [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]],
                dtype=np.float32,
            )

            if not rotate_around_center:
                transformation_matrices[step, ...] = (
                    rotation_matrix @ shift_matrix @ scale_matrix
                )
                continue
            transformation_matrices[step, ...] = (
                centering_matrix
                @ rotation_matrix
                @ inverse_centering_matrix
                @ shift_matrix
                @ scale_matrix
            )

        return transformation_matrices

    def _generate_video_scipy(
        self, image: GrayImage, transformation_matrices: Float[np.ndarray, "T 3 3"]
    ) -> GrayVideo:
        video = np.zeros(
            (transformation_matrices.shape[0], image.shape[0], image.shape[1]),
            dtype=np.float32,
        )
        with tqdm(
            total=len(transformation_matrices), desc="Generating video"
        ) as video_progress_bar:
            for timestep, matrix in enumerate(transformation_matrices):
                video[timestep] = affine_transform(image, matrix=matrix)
                video_progress_bar.update(1)

        return video

    def _generate_video_torch(
        self,
        image: GrayImage,
        angles: List[float],
        translates: List[List[int]],
        scales: List[float],
    ) -> GrayVideo:
        videos = torch.zeros(len(angles), *image.shape).cuda()
        image_tensor = torch.tensor(image)[None, ...].cuda()

        for i, (angle, translate, scale) in enumerate(
            zip(angles, translates, scales)
        ):
            videos[i] = affine(
                image_tensor,
                angle=angle,
                translate=translate,
                shear=[0, 0],
                scale=scale,
            )

        return videos.cpu().numpy()

    def _crop_video(self, video: GrayVideo) -> GrayVideo:
        cropped_video_border = (
            video.shape[1:] - np.array(self.shared_config.target_image_size)
        ) // 2

        cropped_video = video[
            :,
            cropped_video_border[0] : cropped_video_border[0]
            + self.shared_config.target_image_size[0],
            cropped_video_border[1] : cropped_video_border[1]
            + self.shared_config.target_image_size[1],
        ]
        return cropped_video
