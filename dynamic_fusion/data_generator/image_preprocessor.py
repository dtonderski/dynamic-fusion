import logging
from typing import Optional, Tuple

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

    def __init__(self, config: ImagePreprocessorConfiguration, shared_config: SharedConfiguration) -> None:
        self.config = config
        self.shared_config = shared_config
        self.logger = logging.getLogger("ImagePreprocessor")

    def run(self, image: Image) -> Tuple[GrayImageFloat, GrayImageFloat, float]:
        self.logger.info("Preprocessing image...")

        if self.config.max_image_size is not None:
            while np.any(image.shape[:2] > np.array(self.config.max_image_size)):
                self.logger.info(f"Image shape: {image.shape[:2]} larger than max allowed shape {self.config.max_image_size} - halving image size.")
                downscaled_image_size = np.round(np.array(image.shape[:2]) * 0.5)
                image = resize(image, output_shape=downscaled_image_size, order=3, anti_aliasing=True)

        minimum_image_shape = np.array(self.shared_config.minimum_downscaled_image_size) * self.shared_config.downscaling_factor
        if self.shared_config.target_unscaled_image_size is not None:
            minimum_image_shape = np.maximum(minimum_image_shape, np.array(self.shared_config.target_unscaled_image_size))

        if np.any(image.shape[:2] < minimum_image_shape):
            raise ValueError(f"Skipping image - image shape: {image.shape[:2]}, minimum shape: {minimum_image_shape}")

        image = self._rgb2gray(image)
        if not self._validate_contrast(image):
            raise ValueError("Skipping image - low contrast.")
        image = normalize(image)

        if self.config.exponentiate:
            self.logger.debug("Exponentiating image")
            assert self.config.exponentiation_range is not None
            exponentiation_multiplier = float(uniform(low=self.config.exponentiation_range[0], high=self.config.exponentiation_range[1]))
            exponentiated_image = np.exp(image.copy() * exponentiation_multiplier)
            exponentiated_image = normalize(exponentiated_image)
        else:
            exponentiation_multiplier = 1.0
            exponentiated_image = image.copy()

        return image, exponentiated_image, exponentiation_multiplier

    def _rgb2gray(self, image: Image) -> GrayImage:
        if image.ndim > 2 and image.shape[2] > 1:
            image = rgb2gray(image)
        return np.squeeze(image)

    def _validate_contrast(self, image: GrayImage) -> bool:
        return bool((np.max(image) - np.min(image)) > 1.0 / 125)
