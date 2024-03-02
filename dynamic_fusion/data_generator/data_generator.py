import logging

import numpy as np
from jaxtyping import Int32
from tqdm import tqdm
from skimage.transform import resize

from dynamic_fusion.utils.datatypes import GrayVideoFloat
from dynamic_fusion.utils.seeds import set_seeds

from .configuration import DataGeneratorConfiguration
from .data_saver import DataSaver
from .event_discretizer import EventDiscretizer
from .event_generator import EventGenerator
from .image_loader import ImageLoader
from .image_preprocessor import ImagePreprocessor
from .video_generator import VideoGenerator


class DataGenerator:  # pylint: disable=too-many-instance-attributes
    config: DataGeneratorConfiguration
    logger: logging.Logger

    image_loader: ImageLoader
    image_preprocessor: ImagePreprocessor
    video_generator: VideoGenerator
    event_generator: EventGenerator
    event_discretizer: EventDiscretizer
    data_saver: DataSaver
    seeds: Int32[np.ndarray, " N"]

    def __init__(self, config: DataGeneratorConfiguration) -> None:
        self.config = config
        if config.shared.seed is not None:
            set_seeds(config.shared.seed)
        self.seeds = np.random.randint(0, np.iinfo(np.int32).max, self.config.image_loader.number_of_input_images)

        self.logger = logging.getLogger("DataGenerator")

        self.image_loader = ImageLoader(config.image_loader)
        self.image_preprocessor = ImagePreprocessor(config.image_preprocessor, config.shared)
        self.video_generator = VideoGenerator(config.video_generator, config.shared)
        self.event_generator = EventGenerator(config.event_generator, config.shared)
        self.event_discretizer = EventDiscretizer(config.event_discretizer, config.shared)
        self.data_saver = DataSaver(config.data_saver)

    def run(self) -> None:
        image_generator = iter(self.image_loader.run())

        print("-------------------------------------")
        with tqdm(total=len(self.image_loader)) as progress_bar:
            i_image = 0
            while i_image < self.config.image_loader.number_of_input_images:
                set_seeds(self.seeds[i_image])
                progress_bar.set_description(f"Processing image {i_image+1} of {self.config.image_loader.number_of_input_images}")
                print("-------------------------------------")

                image, image_path = next(image_generator)

                if self.data_saver.output_exists(image_path):
                    if self.config.shared.overwrite:
                        self.logger.warning("Output exists but overwrite is true, overwriting!")
                    else:
                        self.logger.info("Output exists and overwrite is false, skipping.")
                        continue
                try:
                    preprocessed_image = self.image_preprocessor.run(image)
                except ValueError as e:
                    self.logger.warning(e)
                    continue

                downscaling_factor = self.config.shared.downscaling_factor
                downscaled_resolution = tuple(int(x / downscaling_factor) for x in preprocessed_image.shape)

                video, downscaled_video, transform_definition = self.video_generator.run(preprocessed_image, downscaled_resolution)  # type: ignore

                unscaled_event_dict = self.event_generator.run(video)
                unscaled_discretized_event_dict = self.event_discretizer.run(unscaled_event_dict, video.shape[1:])

                downscaled_event_dict = self.event_generator.run(downscaled_video, regenerate_luminance=False)
                downscaled_discretized_event_dict = self.event_discretizer.run(downscaled_event_dict, downscaled_video.shape[1:])

                self.data_saver.run(
                    image_path,
                    image,
                    video,
                    downscaled_video,
                    preprocessed_image,
                    transform_definition,
                    unscaled_event_dict,
                    downscaled_event_dict,
                    unscaled_discretized_event_dict,
                    downscaled_discretized_event_dict,
                )
                i_image += 1
                print("-------------------------------------")
