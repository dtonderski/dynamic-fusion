import logging
from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float64, Int64
from tqdm import tqdm

from on_the_fly.trainers.utils.datatypes import (
    DiscretizedEventsStatistics,
    Events,
    EventTensors,
    GrayVideoTorch,
    TemporalBinIndices,
    TimeStamps,
)
from on_the_fly.trainers.utils.discretized_events import DiscretizedEvents

from .configuration import EventDiscretizerConfiguration, SharedConfiguration

ONE = torch.tensor(1.0, dtype=torch.float32)


class EventDiscretizer:
    config: EventDiscretizerConfiguration
    shared_config: SharedConfiguration
    logger: logging.Logger

    def __init__(
        self,
        config: EventDiscretizerConfiguration,
        shared_config: SharedConfiguration,
    ) -> None:
        self.config = config
        self.shared_config = shared_config
        self.logger = logging.getLogger("EventDiscretizer")

    def run(
        self,
        events_dict: Dict[float, Events],
        logarithmic_video: GrayVideoTorch,
        progress_bar: Optional[tqdm] = None,
    ) -> Tuple[Dict[float, DiscretizedEvents], GrayVideoTorch]:
        if progress_bar:
            progress_bar.set_postfix_str("Discretizing events")
        else:
            self.logger.info("Discretizing events...")

        discretized_events_dict = {}
        for threshold, events in events_dict.items():
            discretized_events_dict[threshold] = self._discretize_events(
                events, threshold
            )

        indices_of_label_frames = self._calculate_indices_of_label_frames()

        ground_truth = logarithmic_video[indices_of_label_frames, :, :]
        return discretized_events_dict, ground_truth

    def _calculate_indices_of_label_frames(self) -> Int64[torch.Tensor, " N"]:
        r"""This function calculates the indices of the video frames that are
        the labels of the corresponding discretized events.

        If we have N input video frames, then we have N-1 intervals. If a
        neural network processes a temporal bin that spans interval
        indices [i,j], the output should be the voltage of interval j+1.

        Examples:

            1. 302 input video frames, 6 temporal bins gives:
                301 intervals, discretized_frame_length = 50,
                ground_truth_video_indices = [50, 100, 150, 200, 250, 300].

            2. 8 input video frames, 3 temporal bins gives:
                7 intervals, discretized_frame_length = 2,
                ground_truth_video_indices = [2, 4, 6].
        """

        # Edges of temporal bins must align frames.
        assert (
            self.shared_config.number_of_images_to_generate_per_input - 1
        ) % self.config.number_of_temporal_bins == 0

        discretized_frame_length = (
            self.shared_config.number_of_images_to_generate_per_input - 1
        ) // self.config.number_of_temporal_bins

        return torch.arange(
            discretized_frame_length - 1,
            self.shared_config.number_of_images_to_generate_per_input - 1,
            discretized_frame_length,
            dtype=torch.int64,
        )

    def _discretize_events(
        self, events: Events, threshold: float
    ) -> DiscretizedEvents:
        max_timestamp = (
            self.shared_config.number_of_images_to_generate_per_input - 1
        ) / self.shared_config.fps
        timestamps: TimeStamps = torch.tensor(events.timestamp.values.astype(float))

        events_torch: EventTensors = (
            timestamps,
            torch.tensor(events.x.values.astype(int)),
            torch.tensor(events.y.values.astype(int)),
            torch.tensor(events.polarity.values.astype(bool)),
        )

        temporal_bin_indices = self._calculate_temporal_bin_indices(
            timestamps, max_timestamp
        )

        timestamps_in_bins = (
            (timestamps / max_timestamp * self.config.number_of_temporal_bins)
            - temporal_bin_indices
        ).float()

        resolution = (
            self.config.number_of_temporal_bins,
            *self.shared_config.target_image_size,
        )

        event_polarity_sum: DiscretizedEventsStatistics = (
            self._calculate_event_polarity_sum(
                events_torch, temporal_bin_indices, threshold, resolution
            )
        )

        timestamp_mean, timestamp_std, event_count = self._calculate_statistics(
            events_torch,
            timestamps_in_bins,
            temporal_bin_indices,
            resolution,
        )

        return DiscretizedEvents(
            event_polarity_sum, timestamp_mean, timestamp_std, event_count
        )

    def _calculate_statistics(
        self,
        events_torch: EventTensors,
        timestamps_in_bins: TimeStamps,
        temporal_bin_indices: TemporalBinIndices,
        resolution: Tuple[int, ...],
    ) -> Tuple[
        DiscretizedEventsStatistics,
        DiscretizedEventsStatistics,
        DiscretizedEventsStatistics,
    ]:
        _, x_s, y_s, _ = events_torch

        timestamp_sum: DiscretizedEventsStatistics = torch.zeros(resolution)
        timestamp_squared_sum: DiscretizedEventsStatistics = torch.zeros(resolution)
        timestamp_count: DiscretizedEventsStatistics = torch.zeros(resolution)
        event_count: DiscretizedEventsStatistics = torch.zeros(resolution)
        timestamp_mean: DiscretizedEventsStatistics = torch.zeros(resolution)
        timestamp_std: DiscretizedEventsStatistics = torch.zeros(resolution)

        timestamp_sum.index_put_(
            (temporal_bin_indices, y_s, x_s), timestamps_in_bins, accumulate=True
        )
        timestamp_squared_sum.index_put_(
            (temporal_bin_indices, y_s, x_s), timestamps_in_bins**2, accumulate=True
        )
        timestamp_count.index_put_(
            (temporal_bin_indices, y_s, x_s), ONE, accumulate=True
        )
        event_count.index_put_((temporal_bin_indices, y_s, x_s), ONE, accumulate=True)
        indices_with_events = event_count > 0
        timestamp_mean[indices_with_events] = (
            timestamp_sum[indices_with_events] / event_count[indices_with_events]
        )
        timestamp_std[indices_with_events] = torch.sqrt(
            (
                timestamp_squared_sum[indices_with_events]
                - (
                    timestamp_sum[indices_with_events] ** 2
                    / event_count[indices_with_events]
                )
            )
            / event_count[indices_with_events]
        )

        return timestamp_mean, timestamp_std, event_count

    def _calculate_event_polarity_sum(
        self,
        events_torch: EventTensors,
        temporal_bin_indices: TemporalBinIndices,
        threshold: float,
        resolution: Tuple[int, ...],
    ) -> DiscretizedEventsStatistics:
        _, x_s, y_s, polarities = events_torch

        event_polarity_sum: DiscretizedEventsStatistics = torch.zeros(resolution)
        event_polarity_sum.index_put_(
            (temporal_bin_indices[polarities], y_s[polarities], x_s[polarities]),
            ONE,
            accumulate=True,
        )

        event_polarity_sum.index_put_(
            (temporal_bin_indices[~polarities], y_s[~polarities], x_s[~polarities]),
            -ONE,
            accumulate=True,
        )

        return event_polarity_sum * threshold

    def _calculate_temporal_bin_indices(
        self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float
    ) -> TemporalBinIndices:
        temporal_bin_indices = torch.floor(
            timestamps / max_timestamp * self.config.number_of_temporal_bins
        )

        temporal_bin_indices = torch.clip(
            temporal_bin_indices, 0, self.config.number_of_temporal_bins - 1
        )
        temporal_bin_indices = temporal_bin_indices.long()
        return temporal_bin_indices
