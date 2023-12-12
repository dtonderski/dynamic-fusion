import logging
from pathlib import Path
from typing import Generator, List, Tuple

import imageio.v3 as iio
import numpy as np

from dynamic_fusion.utils.datatypes import Image

from .configuration import ImageLoaderConfiguration


class ImageLoader:
    config: ImageLoaderConfiguration
    logger: logging.Logger
    image_paths: List[Path]

    def __init__(self, config: ImageLoaderConfiguration) -> None:
        self.config = config
        self.logger = logging.getLogger("ImageLoader")
        self._load_image_paths()

    def _load_image_paths(self) -> None:
        self.logger.info("Loading images...")
        image_paths: List[Path] = list(
            self.config.dataset_dir.glob(f"*.{self.config.file_extension}")
        )
        shuffled_files = list(np.random.permutation(np.array(image_paths)))
        self.image_paths = shuffled_files[: self.config.number_of_input_images]

    def run(self) -> Generator[Tuple[Image, Path], None, None]:
        for image_path in self.image_paths:
            self.logger.info(f"Loading image {image_path.name}...")
            yield iio.imread(image_path), image_path

    def __len__(self) -> int:
        return len(self.image_paths)
