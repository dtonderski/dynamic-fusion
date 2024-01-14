import logging
from typing import Optional

import numpy as np
from numpy.random import uniform
from skimage.color import rgb2gray  # pylint: disable=no-name-in-module
from skimage.transform import resize
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import GrayImage, Image, GrayImageFloat

from .configuration import ImagePreprocessorConfiguration, SharedConfiguration
from .utils.image import normalize


class ImagePreprocessor:
    config: ImagePreprocessorConfiguration
    shared_config: SharedConfiguration
    logger: logging.Logger

    def __init__(
        self,
        config: ImagePreprocessorConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        self.logger = logging.getLogger("ImagePreprocessor")

    def run(self, image: Image, pbar: Optional[tqdm] = None) -> GrayImageFloat:
        if pbar:
            pbar.set_postfix_str("Preprocessing image.")
        else:
            self.logger.info("Preprocessing image...")

        if (
            self.shared_config.target_image_size is not None
            and not self._validate_size(image)
        ):
            raise ValueError(
                f"Skipping image - image shape {image.shape[:2]} "
                "smaller than target shape "
                f"{self.shared_config.target_image_size}."
            )

        image = self._downscale_probabilistically(image)
        image = self._rgb2gray(image)
        if not self._validate_contrast(image):
            raise ValueError("Skipping image - low contrast.")
        image = normalize(image)
        return image

    def _validate_size(self, image: Image) -> bool:
        return np.all(
            image.shape[:2] > np.array(self.shared_config.target_image_size)
        )  # type: ignore

    def _downscale_probabilistically(self, image: Image) -> Image:
        image_size = image.shape[:2]
        if np.random.random() > self.config.downscale_probability:
            return image

        scale = uniform(
            low=self.config.downscale_range[0],
            high=self.config.downscale_range[1],
        )

        if self.shared_config.target_image_size is not None:
            if np.min(image_size) > 2 * np.max(self.shared_config.target_image_size):
                self.logger.debug("Downscaling large image by halving its size")
                scale = 0.5

        downscaled_image_size = np.round(np.array(image_size) * scale)

        return resize(image, output_shape=downscaled_image_size, anti_aliasing=True)

    def _rgb2gray(self, image: Image) -> GrayImage:
        if image.ndim > 2 and image.shape[2] > 1:
            image = rgb2gray(image)
        return np.squeeze(image)

    def _validate_contrast(self, image: GrayImage) -> bool:
        return bool(np.max(image) - np.min(image) > 1.0 / 125)
