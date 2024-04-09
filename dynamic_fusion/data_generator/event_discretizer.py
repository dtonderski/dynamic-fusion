import logging
from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float64

from dynamic_fusion.utils.datatypes import DiscretizedEventsStatistics, Events, EventTensors, TemporalBinIndices, TemporalSubBinIndices, TimeStamps
from dynamic_fusion.utils.discretized_events import DiscretizedEvents

from .configuration import EventDiscretizerConfiguration, SharedConfiguration

ONE = torch.tensor(1.0, dtype=torch.float32)


class EventDiscretizer:
    config: EventDiscretizerConfiguration
    max_timestamp: float
    logger: logging.Logger

    def __init__(
        self,
        config: EventDiscretizerConfiguration,
        shared_config: Optional[SharedConfiguration] = None,
        max_timestamp: Optional[float] = None,
    ) -> None:
        self.config = config
        if shared_config is not None:
            self.max_timestamp = (shared_config.number_of_images_to_generate_per_input - 1) / shared_config.fps
        elif max_timestamp is not None:
            self.max_timestamp = max_timestamp
        else:
            raise ValueError("Invalid arguments to EventDiscretizer")

        self.logger = logging.getLogger("EventDiscretizer")

    def run(self, events_dict: Dict[float, Events], image_resolution: Optional[Tuple[int, int]]) -> Dict[float, DiscretizedEvents]:
        self.logger.info("Discretizing events...")

        if image_resolution is None:
            self.logger.warning("Image resolution not given! Infering from events.")
            smallest_threshold_events = min(events_dict.items(), key=lambda x: x[0])[1]
            image_resolution = (smallest_threshold_events.y.max(), smallest_threshold_events.x.max())

        discretized_events_dict = {}
        for threshold, events in events_dict.items():
            discretized_events_dict[threshold] = self._discretize_events(events, threshold, image_resolution)

        return discretized_events_dict

    def _discretize_events(
        self,
        events: Events,
        threshold: float,
        image_resolution: Tuple[int, int],
    ) -> DiscretizedEvents:
        timestamps: TimeStamps = torch.tensor(events.timestamp.values.astype(float))

        events_torch: EventTensors = (
            timestamps,
            torch.tensor(events.x.values.astype(int)),
            torch.tensor(events.y.values.astype(int)),
            torch.tensor(events.polarity.values.astype(bool)),
        )

        temporal_bin_indices = self._calculate_temporal_bin_indices(timestamps, self.max_timestamp)
        temporal_sub_bin_indices = self._calculate_temporal_sub_bin_indices(timestamps, self.max_timestamp)

        # Normalize so they lie between [0, n_bins*sub_bins_per_bin]
        normalized_timestamps = timestamps / self.max_timestamp * self.config.number_of_temporal_bins * self.config.number_of_temporal_sub_bins_per_bin

        # Normalize so events in each sub-bin lie between [0, 1]
        timestamps_in_sub_bins = (
            normalized_timestamps - temporal_sub_bin_indices - temporal_bin_indices * self.config.number_of_temporal_sub_bins_per_bin
        ).float()

        resolution = (self.config.number_of_temporal_bins, self.config.number_of_temporal_sub_bins_per_bin, *image_resolution)

        event_polarity_sum: DiscretizedEventsStatistics = self._calculate_event_polarity_sum(
            events_torch,
            temporal_bin_indices,
            temporal_sub_bin_indices,
            threshold,
            resolution,
        )

        timestamp_mean, timestamp_std, event_count = self._calculate_statistics(
            events_torch,
            timestamps_in_sub_bins,
            temporal_bin_indices,
            temporal_sub_bin_indices,
            resolution,
        )

        return DiscretizedEvents(event_polarity_sum, timestamp_mean, timestamp_std, event_count)

    @staticmethod
    def _calculate_statistics(
        events_torch: EventTensors,
        timestamps_in_sub_bins: TimeStamps,
        temporal_bin_indices: TemporalBinIndices,
        temporal_sub_bin_indices: TemporalSubBinIndices,
        resolution: Tuple[int, int, int, int],
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
            (temporal_bin_indices, temporal_sub_bin_indices, y_s, x_s),
            timestamps_in_sub_bins,
            accumulate=True,
        )
        timestamp_squared_sum.index_put_(
            (temporal_bin_indices, temporal_sub_bin_indices, y_s, x_s),
            timestamps_in_sub_bins**2,
            accumulate=True,
        )
        timestamp_count.index_put_(
            (temporal_bin_indices, temporal_sub_bin_indices, y_s, x_s),
            ONE,
            accumulate=True,
        )
        event_count.index_put_(
            (temporal_bin_indices, temporal_sub_bin_indices, y_s, x_s),
            ONE,
            accumulate=True,
        )

        indices_with_events = event_count > 0
        timestamp_mean[indices_with_events] = timestamp_sum[indices_with_events] / event_count[indices_with_events]

        numerator = timestamp_squared_sum[indices_with_events] - (timestamp_sum[indices_with_events] ** 2 / event_count[indices_with_events])
        variance = numerator / event_count[indices_with_events]

        # Can be negative for extremely close timestamps (for example in the case of dead pixels)
        variance = torch.clamp(variance, min=0)

        timestamp_std[indices_with_events] = torch.sqrt(variance)

        return timestamp_mean, timestamp_std, event_count

    @staticmethod
    def _calculate_event_polarity_sum(
        events_torch: EventTensors,
        temporal_bin_indices: TemporalBinIndices,
        temporal_sub_bin_indices: TemporalSubBinIndices,
        threshold: float,
        resolution: Tuple[int, int, int, int],  # T D H W
    ) -> DiscretizedEventsStatistics:
        _, x_s, y_s, polarities = events_torch

        event_polarity_sum: DiscretizedEventsStatistics = torch.zeros(resolution)
        event_polarity_sum.index_put_(
            (
                temporal_bin_indices[polarities],
                temporal_sub_bin_indices[polarities],
                y_s[polarities],
                x_s[polarities],
            ),
            ONE,
            accumulate=True,
        )

        event_polarity_sum.index_put_(
            (
                temporal_bin_indices[~polarities],
                temporal_sub_bin_indices[~polarities],
                y_s[~polarities],
                x_s[~polarities],
            ),
            -ONE,
            accumulate=True,
        )

        return event_polarity_sum * threshold

    def _calculate_temporal_bin_indices(self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float) -> TemporalBinIndices:
        temporal_bin_indices = torch.floor(timestamps / max_timestamp * self.config.number_of_temporal_bins)

        temporal_bin_indices = torch.clip(temporal_bin_indices, 0, self.config.number_of_temporal_bins - 1)
        temporal_bin_indices = temporal_bin_indices.long()
        return temporal_bin_indices

    def _calculate_temporal_sub_bin_indices(self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float) -> TemporalBinIndices:
        total_sub_bin_indices = torch.floor(timestamps / max_timestamp * self.config.number_of_temporal_bins * self.config.number_of_temporal_sub_bins_per_bin)

        sub_bin_indices_in_bins = torch.remainder(total_sub_bin_indices, self.config.number_of_temporal_sub_bins_per_bin)
        sub_bin_indices_in_bins = torch.clip(sub_bin_indices_in_bins, 0, self.config.number_of_temporal_sub_bins_per_bin - 1)
        sub_bin_indices_in_bins = sub_bin_indices_in_bins.long()
        return sub_bin_indices_in_bins
