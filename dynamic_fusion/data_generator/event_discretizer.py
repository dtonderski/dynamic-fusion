import logging
from abc import ABC, abstractmethod
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

    def run(self, events_dict: Dict[float, Events], image_resolution: Tuple[int, int]) -> Dict[float, DiscretizedEvents]:
        self.logger.info("Discretizing events...")

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

        discretization_type = self.config.discretization_type
        if discretization_type == 'fixed_bin_num':
            temporal_bin_indices = self._calculate_temporal_bin_indices(timestamps, self.max_timestamp)
            temporal_sub_bin_indices = self._calculate_temporal_sub_bin_indices(timestamps, self.max_timestamp)
        elif discretization_type == 'fixed_evexitent_num':
            events_torch, temporal_bin_indices, temporal_sub_bin_indices = self._prepare_inputs_fixed_event_num(
                events_torch,
                self.config.number_of_temporal_bins,
                self.config.number_of_events_per_bin,
                self.config.number_of_temporal_sub_bins_per_bin,
                )
        else:
            self.logger.warning(f"Invalid parameter discretization_type = {discretization_type}.")

        # Normalize so they lie between [0, n_bins*sub_bins_per_bin]
        normalized_timestamps = timestamps / self.max_timestamp * self.config.number_of_temporal_bins * self.config.number_of_temporal_sub_bins_per_bin

        # Normalize so events in each sub-bin lie between [0, 1]
        timestamps_in_sub_bins = (
            normalized_timestamps - temporal_sub_bin_indices - temporal_bin_indices * self.config.number_of_temporal_sub_bins_per_bin
        ).float()

        resolution = (
            self.config.number_of_temporal_bins,
            self.config.number_of_temporal_sub_bins_per_bin,
            *image_resolution,
        )

        
        event_polarity_sum: DiscretizedEventsStatistics = self._calculate_event_polarity_sum(
            events_torch,
            temporal_bin_indices,
            temporal_sub_bin_indices,
            threshold,
            resolution,
            self.config.approximation_type,
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
        timestamp_std[indices_with_events] = torch.sqrt(
            (timestamp_squared_sum[indices_with_events] - (timestamp_sum[indices_with_events] ** 2 / event_count[indices_with_events]))
            / event_count[indices_with_events]
        )

        return timestamp_mean, timestamp_std, event_count

    @staticmethod
    def _calculate_event_polarity_sum(
        events_torch: EventTensors,
        temporal_bin_indices: TemporalBinIndices,
        temporal_sub_bin_indices: TemporalSubBinIndices,
        threshold: float,
        resolution: Tuple[int, int, int, int],  # T D H W
        approximation_type: str,
    ) -> DiscretizedEventsStatistics:
        _, x_s, y_s, polarities = events_torch

        # TODO: add support for spatio-temporal voxel grid approximation (as in e2vid paper)
        if approximation_type == 'nearest':
            p_s = polarities * threshold + ~polarities * (-threshold)
        else:
            raise Exception(f"Invalid parameter approximation_type = {approximation_type}.")

        event_polarity_sum: DiscretizedEventsStatistics = torch.zeros(resolution)
        event_polarity_sum.index_put_(
            (
                temporal_bin_indices,
                temporal_sub_bin_indices,
                y_s,
                x_s,
            ),
            p_s,
            accumulate=True,
        )

        return event_polarity_sum

    def _calculate_temporal_bin_indices(self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float) -> TemporalBinIndices:
        temporal_bin_indices = torch.floor(timestamps / max_timestamp * self.config.number_of_temporal_bins)

        temporal_bin_indices = torch.clip(temporal_bin_indices, 0, self.config.number_of_temporal_bins - 1)
        temporal_bin_indices = temporal_bin_indices.long()
        return temporal_bin_indices

    def _calculate_temporal_sub_bin_indices(self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float) -> TemporalBinIndices:
        total_sub_bin_indices = torch.floor(timestamps / max_timestamp * self.config.number_of_temporal_bins * self.config.number_of_temporal_sub_bins_per_bin)

        sub_bin_indices_in_bins = torch.remainder(total_sub_bin_indices, self.config.number_of_temporal_sub_bins_per_bin)
        sub_bin_indices_in_bins = torch.clip(
            sub_bin_indices_in_bins,
            0,
            self.config.number_of_temporal_sub_bins_per_bin - 1,
        )
        sub_bin_indices_in_bins = sub_bin_indices_in_bins.long()
        return sub_bin_indices_in_bins

    def _prepare_inputs_fixed_event_num(
            self,
            events_torch,
            number_of_bins,
            number_of_events_per_bin,
            number_of_sub_bins_per_bin,
            ):
        timestamps, x_s, y_s, polarities = events_torch

        required_num_events = number_of_events_per_bin * number_of_bins
        if required_num_events > self.max_timestamp:
            self.logger.warning(
                f"Not enough number of generated events, need {required_num_events}, got max {self.max_timestamp}."
                )

        # Ensure correct number of events
        if timestamps.shape[0] < required_num_events:
            timestamps = timestamps[:required_num_events]
            x_s = x_s[:required_num_events]
            y_s = y_s[:required_num_events]
            polarities = polarities[:required_num_events]

        # Create temporal bin indices
        temporal_bin_indices = torch.arange(number_of_bins).repeat_interleave(number_of_events_per_bin)

        # Calculate sub-bin indices in bins
        time_bin_matrix = timestamps.view(number_of_bins, number_of_events_per_bin)
        t_start = time_bin_matrix[:, 0]
        t_finish = time_bin_matrix[:, -1]
        interval_lens = (t_finish - t_start).view(number_of_bins, 1)
        normilized_time = (time_bin_matrix - t_start.view(number_of_bins, 1)) / interval_lens * (number_of_sub_bins_per_bin - 1)
        sub_bin_indices_in_bins = torch.round(normilized_time).long().flatten()

        return (timestamps, x_s, y_s, polarities), temporal_bin_indices, sub_bin_indices_in_bins
