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
            for i_image in tqdm(range(len(self.image_loader))):
                set_seeds(self.seeds[i_image])
                progress_bar.set_description(f"Processing image {i_image+1} of {len(self.image_loader)}")
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

                video, transform_definition = self.video_generator.run(preprocessed_image)

                downscaling_factor = np.random.random() * (self.config.max_downscaling_factor - 1) + 1
                downscaled_image_size = np.round(np.array(video.shape[1:]) / downscaling_factor)
                downscaled_video = np.stack(
                    [resize(video_frame, output_shape=downscaled_image_size, order=3, anti_aliasing=True) for video_frame in video]
                )

                event_dict = self.event_generator.run(video)
                downscaled_event_dict = self.event_generator.run(downscaled_video, regenerate_luminance=False)

                discretized_event_dict, indices_of_label_frames = self.event_discretizer.run(event_dict, video.shape[1:])
                downscaled_discretized_event_dict, _ = self.event_discretizer.run(downscaled_event_dict, downscaled_video.shape[1:])

                ground_truth_video: GrayVideoFloat = video[indices_of_label_frames, :, :]

                self.data_saver.run(
                    image_path,
                    image,
                    video,
                    downscaled_video,
                    preprocessed_image,
                    transform_definition,
                    event_dict,
                    downscaled_event_dict,
                    discretized_event_dict,
                    downscaled_discretized_event_dict,
                    ground_truth_video,
                )
                print("-------------------------------------")
