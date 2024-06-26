import logging
from typing import Dict, Generator, Optional, Tuple

import evs_explorer
import numpy as np
import ruamel.yaml
from evs_explorer.configuration import ContrastStep
from evs_explorer.configuration.utils import yaml
from evs_explorer.pipeline import InputNode
from jaxtyping import UInt8
from numpy.random import uniform
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import Events, GrayVideoFloat

from .configuration import EventGeneratorConfiguration, SharedConfiguration


# This class is used as an input node which can read in a sequence of images
# and then feed them into the simulator one-by-one at each timestep.
@yaml.register_class
class ImageGenerator(InputNode):  # type: ignore
    """
    Modified version of evs_explorer.pipeline.GeneratorNode so that it can be
    specified as an input node.
    """

    yaml_tag = "!ImageGenerator"

    def __init__(self, data: UInt8[np.ndarray, "T H W"], fps: float, num_frames: int):
        super().__init__(fps=fps, num_frames=num_frames, shape=data.shape[1:])

        self.gen = data

    @classmethod
    def to_yaml(cls, representer: ruamel.yaml.Representer, data: "ImageGenerator"):  # type: ignore
        return representer.represent_mapping(
            cls.yaml_tag,
            {
                "fps": data.fps,
                "num_frames": data.num_frames,
            },
        )

    def run(self) -> Generator[UInt8[np.ndarray, "H W"], None, None]:
        yield from self.gen


class EventGenerator:
    config: EventGeneratorConfiguration
    shared_config: SharedConfiguration
    evs_config: evs_explorer.Configuration
    logger: logging.Logger

    def __init__(
        self,
        config: EventGeneratorConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        self.evs_config = evs_explorer.Configuration.from_yaml(
            simulator_config=self.config.simulator_config_path,
            sensor_config=self.config.sensor_config_path,
        )  # pyright: ignore
        self.logger = logging.getLogger("EventGenerator")

    def run(
        self, video: GrayVideoFloat, regenerate_luminance: bool = True, illuminance_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[Dict[float, Events], Tuple[float, float]]:
        self.logger.info("Generating events...")
        if illuminance_range is not None:
            self.evs_config.optics.max_illuminance_lux = illuminance_range[1]
            self.evs_config.optics.min_illuminance_lux = illuminance_range[0]
        elif regenerate_luminance:
            illuminance_range = self._generate_luminance()
            # Must be in this order or evs complains about max < min
            self.evs_config.optics.max_illuminance_lux = illuminance_range[1]
            self.evs_config.optics.min_illuminance_lux = illuminance_range[0]
        else:
            illuminance_range = (self.evs_config.optics.min_illuminance_lux, self.evs_config.optics.max_illuminance_lux)

        sensor_data = {}

        video = video * 255

        image_generator = ImageGenerator(data=video, fps=self.shared_config.fps, num_frames=len(video))

        self.evs_config.input.source = image_generator

        for threshold in self.config.thresholds:
            self.logger.info(f"Generating events for threshold {threshold}...")
            sensor_data[threshold] = self._run_one_threshold(threshold)
        return sensor_data, illuminance_range

    def _generate_luminance(self) -> Tuple[float, float]:
        while True:
            min_illuminance_lux = uniform(
                low=self.config.min_illuminance_lux_range[0],
                high=self.config.min_illuminance_lux_range[1],
            )

            max_illuminance_lux = uniform(
                low=self.config.max_illuminance_lux_range[0],
                high=self.config.max_illuminance_lux_range[1],
            )

            if max_illuminance_lux - min_illuminance_lux > 50 and max_illuminance_lux / min_illuminance_lux > 4.5:
                break

        return min_illuminance_lux, max_illuminance_lux

    def _run_one_threshold(self, threshold: float) -> Events:
        self._set_threshold(threshold)

        evs = evs_explorer.EvsExplorer(self.evs_config)

        return evs.run("sensor_data")  # type: ignore

    def _set_threshold(self, threshold: float) -> None:
        if self.evs_config.sensor_model == "davis_model":
            self.evs_config.sensor.ONth_mul = threshold
            self.evs_config.sensor.OFFth_mul = threshold
        elif self.evs_config.sensor_model == "sees_model":
            self.evs_config.sensor.set_sensor_event_sensitivity(ContrastStep(threshold))  # pyright: ignore
