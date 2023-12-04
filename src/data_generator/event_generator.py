import copy
import logging
from typing import Dict, List, Optional, Tuple

import evs_explorer
import torch
from evs_explorer.configuration import ContrastStep
from numpy.random import uniform
from tqdm import tqdm

from on_the_fly.simulators.evs_explorer_interface import ImageGenerator
from on_the_fly.trainers.utils.datatypes import Events, GrayVideo, GrayVideoTorch

from .configuration import EventGeneratorConfiguration, SharedConfiguration


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
        self, video: GrayVideo, progress_bar: Optional[tqdm] = None
    ) -> Tuple[Dict[float, Events], GrayVideoTorch]:
        if progress_bar:
            progress_bar.set_postfix_str("Generating events")
        else:
            self.logger.info("Generating events...")

        self._update_config()
        sensor_data = {}

        video = video * 255

        image_generator = ImageGenerator(
            data=video, fps=self.shared_config.fps, num_frames=len(video)
        )

        self.evs_config.input.source = image_generator
        self.logger.info("Generating ground truth images...")
        logarithmic_video = self._generate_logarithmic_video()

        for threshold in self.config.thresholds:
            self.logger.info(f"Generating events for threshold {threshold}...")
            sensor_data[threshold] = self._run_one_threshold(threshold)
        return sensor_data, logarithmic_video

    def _update_config(self) -> None:
        min_illuminance_lux, max_illuminance_lux = self._generate_luminance()
        # Must be in this order or evs complains about max < min
        self.evs_config.optics.max_illuminance_lux = max_illuminance_lux
        self.evs_config.optics.min_illuminance_lux = min_illuminance_lux

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

            if (
                max_illuminance_lux - min_illuminance_lux > 50
                and max_illuminance_lux / min_illuminance_lux > 4.5
            ):
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
            self.evs_config.sensor.set_sensor_event_sensitivity(  # pyright: ignore
                ContrastStep(threshold)
            )

    def _generate_logarithmic_video(self) -> GrayVideoTorch:
        self._set_threshold(5.0)
        clean_evs_config = copy.deepcopy(self.evs_config)
        clean_evs_config.optics.enabled = True
        clean_evs_config.sensor.noise = False
        clean_evs_config.sensor.background_activity = False
        clean_evs_config.sensor.light_shot_noise = False
        clean_evs_config.sensor.mismatch = False
        clean_evs_config.sensor.random_state = False

        evs = evs_explorer.EvsExplorer(clean_evs_config)

        sensor_states: List[Dict[str, torch.Tensor]] = evs.run(  # pyright: ignore
            evs.sensor_node.output("sensor_state")
        )
        voltages = [sensor_state["vpr_prev_V"] for sensor_state in sensor_states]

        return torch.stack(voltages)
