import logging

from tqdm import tqdm

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

    def __init__(self, config: DataGeneratorConfiguration) -> None:
        self.config = config
        if config.shared.seed is not None:
            set_seeds(config.shared.seed)

        self.logger = logging.getLogger("DataGenerator")

        self.image_loader = ImageLoader(config.image_loader)

        self.image_preprocessor = ImagePreprocessor(
            config.image_preprocessor, config.shared
        )

        self.video_generator = VideoGenerator(config.video_generator, config.shared)

        self.event_generator = EventGenerator(config.event_generator, config.shared)

        self.event_discretizer = EventDiscretizer(
            config.event_discretizer, config.shared
        )

        self.data_saver = DataSaver(config.data_saver)

    def run(self) -> None:
        # Pseudocode
        image_generator = iter(self.image_loader.run())

        print("-------------------------------------")
        with tqdm(total=len(self.image_loader)) as progress_bar:
            for i_image in tqdm(range(len(self.image_loader))):
                progress_bar.set_description(
                    f"Processing image {i_image+1} of {len(self.image_loader)}"
                )
                print("-------------------------------------")

                image, image_path = next(image_generator)

                if self.data_saver.output_exists(image_path):
                    if self.config.shared.overwrite:
                        self.logger.warning(
                            "Output exists but overwrite is true, overwriting!"
                        )
                    else:
                        self.logger.info(
                            "Output exists and overwrite is false, skipping."
                        )
                        continue

                preprocessed_image = self.image_preprocessor.run(image)

                video = self.video_generator.run(preprocessed_image)

                event_dict, logarithmic_video = self.event_generator.run(video)

                discretized_event_dict, synchronized_logarithmic_video = (
                    self.event_discretizer.run(event_dict, logarithmic_video)
                )

                self.data_saver.run(
                    image_path,
                    image,
                    video,
                    discretized_event_dict,
                    synchronized_logarithmic_video,
                )
                print("-------------------------------------")
