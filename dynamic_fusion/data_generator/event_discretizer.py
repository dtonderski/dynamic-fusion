import logging
from typing import Dict, List, Optional, Tuple

import torch
from jaxtyping import Float64, Int64
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import (
    DiscretizedEventsStatistics,
    Events,
    EventTensors,
    GrayVideoTorch,
    TemporalBinIndices,
    TemporalSubBinIndices,
    TimeStamps,
)
from dynamic_fusion.utils.discretized_events import DiscretizedEvents

from .configuration import EventDiscretizerConfiguration, SharedConfiguration

ONE = torch.tensor(1.0, dtype=torch.float32)


class EventDiscretizer:
    config: EventDiscretizerConfiguration
    number_of_images_to_generate_per_input: int
    fps: int
    target_image_size: Optional[Tuple[int, int]]
    logger: logging.Logger

    def __init__(
        self,
        config: EventDiscretizerConfiguration,
        shared_config: Optional[SharedConfiguration] = None,
        number_of_images_to_generate_per_input: Optional[int] = None,
        fps: Optional[int] = None,
        target_image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.config = config
        if shared_config is not None:
            self.number_of_images_to_generate_per_input = (
                shared_config.number_of_images_to_generate_per_input
            )
            self.fps = shared_config.fps
            self.target_image_size = shared_config.target_image_size
        elif (
            number_of_images_to_generate_per_input is not None
            and fps is not None
            and target_image_size is not None
        ):
            self.number_of_images_to_generate_per_input = (
                number_of_images_to_generate_per_input
            )
            self.fps = fps
            self.target_image_size = target_image_size
        else:
            raise ValueError("Invalid arguments to EventDiscretizer")

        self.logger = logging.getLogger("EventDiscretizer")

    def run(
        self,
        events_dict: Dict[float, Events],
        image_resolution: Optional[Tuple[int, int]] = None,
        progress_bar: Optional[tqdm] = None,
    ) -> Tuple[Dict[float, DiscretizedEvents], Int64[torch.Tensor, " N"]]:
        if progress_bar:
            progress_bar.set_postfix_str("Discretizing events")
        else:
            self.logger.info("Discretizing events...")

        if self.target_image_size is None and image_resolution is None:
            raise ValueError("Resolution must be set if target_image_size is None!")

        discretized_events_dict = {}
        for threshold, events in events_dict.items():
            discretized_events_dict[threshold] = self._discretize_events(
                events, threshold, image_resolution
            )

        indices_of_label_frames = self._calculate_indices_of_label_frames()

        return discretized_events_dict, indices_of_label_frames

    def _calculate_indices_of_label_frames(self) -> Int64[torch.Tensor, " N"]:
        r"""This function calculates the indices of the video frames that are
        the labels of the corresponding discretized events.

        If we have N input video frames, then we have N-1 intervals. If a
        neural network processes a temporal bin that spans interval
        indices [i,j], the output should be the voltage of interval j+1.

        Examples:

            1. 301 input video frames, 6 temporal bins gives:
                300 intervals, discretized_frame_length = 50,
                ground_truth_video_indices = [50, 100, 150, 200, 250, 300].

            2. 7 input video frames, 3 temporal bins gives:
                6 intervals, discretized_frame_length = 2,
                ground_truth_video_indices = [2, 4, 6].
        """

        # Edges of temporal bins must align frames.
        assert (
            self.number_of_images_to_generate_per_input - 1
        ) % self.config.number_of_temporal_bins == 0

        discretized_frame_length = (
            self.number_of_images_to_generate_per_input - 1
        ) // self.config.number_of_temporal_bins

        return torch.arange(
            discretized_frame_length,
            self.number_of_images_to_generate_per_input - 1,
            discretized_frame_length,
            dtype=torch.int64,
        )

    def _discretize_events(
        self,
        events: Events,
        threshold: float,
        image_resolution: Optional[Tuple[int, int]] = None,
    ) -> DiscretizedEvents:
        max_timestamp = (self.number_of_images_to_generate_per_input - 1) / self.fps
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
        temporal_sub_bin_indices = self._calculate_temporal_sub_bin_indices(
            timestamps, max_timestamp
        )

        # Normalize so they lie between [0, n_bins*sub_bins_per_bin]
        normalized_timestamps = (
            timestamps
            / max_timestamp
            * self.config.number_of_temporal_bins
            * self.config.number_of_temporal_sub_bins_per_bin
        )

        # Normalize so events in each sub-bin lie between [0, 1]
        timestamps_in_sub_bins = (
            normalized_timestamps
            - temporal_sub_bin_indices
            - temporal_bin_indices * self.config.number_of_temporal_sub_bins_per_bin
        ).float()

        if self.target_image_size is not None:
            image_resolution = self.target_image_size
        
        if image_resolution is None:
            raise ValueError("Image resolution is None!")

        resolution = (
            self.config.number_of_temporal_bins,
            self.config.number_of_temporal_sub_bins_per_bin,
            *image_resolution,
        )

        event_polarity_sum: DiscretizedEventsStatistics = (
            self._calculate_event_polarity_sum(
                events_torch,
                temporal_bin_indices,
                temporal_sub_bin_indices,
                threshold,
                resolution,
            )
        )

        timestamp_mean, timestamp_std, event_count = self._calculate_statistics(
            events_torch,
            timestamps_in_sub_bins,
            temporal_bin_indices,
            temporal_sub_bin_indices,
            resolution,
        )

        return DiscretizedEvents(
            event_polarity_sum, timestamp_mean, timestamp_std, event_count
        )

    def _calculate_statistics(
        self,
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

    def _calculate_temporal_sub_bin_indices(
        self, timestamps: Float64[torch.Tensor, " N"], max_timestamp: float
    ) -> TemporalBinIndices:
        total_sub_bin_indices = torch.floor(
            timestamps
            / max_timestamp
            * self.config.number_of_temporal_bins
            * self.config.number_of_temporal_sub_bins_per_bin
        )

        sub_bin_indices_in_bins = torch.remainder(
            total_sub_bin_indices, self.config.number_of_temporal_sub_bins_per_bin
        )
        sub_bin_indices_in_bins = torch.clip(
            sub_bin_indices_in_bins,
            0,
            self.config.number_of_temporal_sub_bins_per_bin - 1,
        )
        sub_bin_indices_in_bins = sub_bin_indices_in_bins.long()
        return sub_bin_indices_in_bins
