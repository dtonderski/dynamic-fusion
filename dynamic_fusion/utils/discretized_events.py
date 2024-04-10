from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import h5py
import numpy as np
import torch
from jaxtyping import Float

from dynamic_fusion.utils.network import to_numpy
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap


@dataclass
class DiscretizedEvents:
    event_polarity_sum: Float[torch.Tensor, "T D X Y"]
    timestamp_mean: Float[torch.Tensor, "T D X Y"]
    timestamp_std: Float[torch.Tensor, "T D X Y"]
    event_count: Float[torch.Tensor, "T D X Y"]

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

    def get_colored_polarity(self) -> Float[torch.Tensor, "T X Y 3"]:
        return img_to_colormap(to_numpy(self.event_polarity_sum.sum(dim=1)), create_red_blue_cmap(501))

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
            event_polarity_sum=torch.from_numpy(np.array(file["event_polarity_sum"][t_start:t_end, x_start:x_end, y_start:y_end])),
            timestamp_mean=torch.from_numpy(np.array(file["timestamp_mean"][t_start:t_end, x_start:x_end, y_start:y_end])),
            timestamp_std=torch.from_numpy(np.array(file["timestamp_std"][t_start:t_end, x_start:x_end, y_start:y_end])),
            event_count=torch.from_numpy(np.array(file["event_count"][t_start:t_end, x_start:x_end, y_start:y_end])),
        )

    @classmethod
    def stack_temporally(cls, discretized_frames: List[DiscretizedEvents]) -> DiscretizedEvents:
        return DiscretizedEvents(
            event_polarity_sum=torch.concat([x.event_polarity_sum for x in discretized_frames], dim=0),
            timestamp_mean=torch.concat([x.timestamp_mean for x in discretized_frames], dim=0),
            timestamp_std=torch.concat([x.timestamp_std for x in discretized_frames], dim=0),
            event_count=torch.concat([x.event_count for x in discretized_frames], dim=0),
        )
