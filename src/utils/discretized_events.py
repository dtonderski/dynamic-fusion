from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch

from on_the_fly.trainers.utils.datatypes import DiscretizedEventsStatistics


@dataclass
class DiscretizedEvents:
    event_polarity_sum: DiscretizedEventsStatistics
    timestamp_mean: DiscretizedEventsStatistics
    timestamp_std: DiscretizedEventsStatistics
    event_count: DiscretizedEventsStatistics

    def save_to_file(self, file: h5py.File, h5_compression: int) -> None:
        file.create_dataset(
            "/event_polarity_sum",
            data=self.event_polarity_sum.cpu(),
            compression="gzip",
            compression_opts=h5_compression,
        )
        file.create_dataset(
            "/timestamp_mean",
            data=self.timestamp_mean.cpu(),
            compression="gzip",
            compression_opts=h5_compression,
        )
        file.create_dataset(
            "/timestamp_std",
            data=self.timestamp_std.cpu(),
            compression="gzip",
            compression_opts=h5_compression,
        )
        file.create_dataset(
            "/event_count",
            data=self.event_count.cpu(),
            compression="gzip",
            compression_opts=h5_compression,
        )

    @classmethod
    def load_from_file(
        cls,
        file: h5py.File,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        x_start: Optional[int] = None,
        x_end: Optional[int] = None,
        y_start: Optional[int] = None,
        y_end: Optional[int] = None,
    ) -> DiscretizedEvents:
        return cls(
            event_polarity_sum=torch.from_numpy(
                np.array(
                    file["event_polarity_sum"][
                        t_start:t_end, x_start:x_end, y_start:y_end
                    ]
                )
            ),
            timestamp_mean=torch.from_numpy(
                np.array(
                    file["timestamp_mean"][
                        t_start:t_end, x_start:x_end, y_start:y_end
                    ]
                )
            ),
            timestamp_std=torch.from_numpy(
                np.array(
                    file["timestamp_std"][t_start:t_end, x_start:x_end, y_start:y_end]
                )
            ),
            event_count=torch.from_numpy(
                np.array(
                    file["event_count"][t_start:t_end, x_start:x_end, y_start:y_end]
                )
            ),
        )

    def crop_tensors_in_time(self, start: int, end: int) -> None:
        self.event_polarity_sum = self.event_polarity_sum[start:end]
        self.timestamp_mean = self.timestamp_mean[start:end]
        self.timestamp_std = self.timestamp_std[start:end]
        self.event_count = self.event_count[start:end]
