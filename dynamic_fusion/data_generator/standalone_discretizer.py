import logging
from typing import List, Optional

import h5py
import pandas as pd
from tqdm import tqdm

from .configuration import DataGeneratorConfiguration, EventDiscretizerConfiguration
from .event_discretizer import EventDiscretizer


class StandaloneDiscretizer:
    """(Re-)discretizes given thresholds"""

    logger: logging.Logger

    original_config: DataGeneratorConfiguration
    thresholds: List[float]
    event_discretizer: EventDiscretizer
    allow_overwrite: bool

    def __init__(
        self,
        thresholds: List[float],
        original_config: DataGeneratorConfiguration,
        allow_overwrite: bool = False,
        discretizer_config: Optional[EventDiscretizerConfiguration] = None,
    ) -> None:
        self.logger = logging.getLogger("DataGenerator")
        self.thresholds = thresholds
        discretizer_config = discretizer_config if discretizer_config is not None else original_config.event_discretizer
        self.event_discretizer = EventDiscretizer(discretizer_config, original_config.shared)
        self.original_config = original_config
        self.allow_overwrite = allow_overwrite

    def run(self) -> None:
        existing_output_dirs = list(self.original_config.data_saver.output_dir.glob("*/**"))
        print(f"Found {len(existing_output_dirs)=}")
        for output_dir in tqdm(existing_output_dirs):
            for threshold in self.thresholds:
                files = [output_dir / f"discretized_events_{threshold}.h5", output_dir / f"downscaled_discretized_events_{threshold}.h5"]
                for file in files:
                    if file.exists() and not self.allow_overwrite:
                        raise ValueError(f"Not allowed to overwrite file {file}!")

        for output_dir in tqdm(existing_output_dirs):
            unscaled_event_dict = {threshold: pd.read_hdf(output_dir / f"events_{threshold}.h5") for threshold in self.thresholds}
            downscaled_event_dict = {threshold: pd.read_hdf(output_dir / f"downscaled_events_{threshold}.h5") for threshold in self.thresholds}

            unscaled_discretized_event_dict = self.event_discretizer.run(unscaled_event_dict, None)
            downscaled_discretized_event_dict = self.event_discretizer.run(downscaled_event_dict, None)

            for threshold, discretized_events in unscaled_discretized_event_dict.items():
                with h5py.File(output_dir / f"discretized_events_{threshold}.h5", "w") as file:
                    discretized_events.save_to_file(file, self.original_config.data_saver.h5_compression)

            for threshold, discretized_events in downscaled_discretized_event_dict.items():
                with h5py.File(output_dir / f"downscaled_discretized_events_{threshold}.h5", "w") as file:
                    discretized_events.save_to_file(file, self.original_config.data_saver.h5_compression)
