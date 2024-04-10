from typing import Optional, Tuple

import cv2
import numpy as np
from jaxtyping import Float, UInt8
from matplotlib.axes import Axes

from dynamic_fusion.utils.array import to_numpy

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (255, 255, 255)  # White color
LINE_TYPE = 1
POSITIONS = {0: (10, 10), 1: (10, 30), 2: (10, 50)}  # Position of the text (top left corner)


def add_text_at_row(
    frame: UInt8[np.ndarray, "X Y 3"],
    text: str,
    row: int,
    font_scale: float = FONT_SCALE,
    font: int = FONT,
    font_color: Tuple[int, int, int] = FONT_COLOR,
    line_type: int = LINE_TYPE,
    override_position: Optional[Tuple[int, int]] = None,
) -> None:
    if override_position is not None:
        position = override_position
    else:
        position = POSITIONS[row]
    cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


def add_discretized_events(
    axes: Axes, colored_event_polarity_sum: Float[np.ndarray, "X Y 3"], event_count: Optional[Float[np.ndarray, "B X Y"]] = None
) -> None:
    colored_eps = to_numpy(colored_event_polarity_sum)
    if event_count is not None:
        count = to_numpy(event_count)

    # Calculate EPPF
    eppf = count.sum(axis=0).mean() if event_count is not None else "unknown"

    axes.imshow(colored_eps)
    axes.set_title(f"Events, EPPF={eppf:.2f}")
    axes.axis("off")


def log_std_to_cv2_frame(log_std: Float[np.ndarray, "X Y"], min_std: float = 0.05, max_std: float = 0.5) -> UInt8[np.ndarray, "X Y 3"]:
    log_std = to_numpy(log_std)
    min_log, max_log = np.log(min_std), np.log(max_std)

    clipped_log_std = np.clip(log_std, min_log, max_log)
    normalized_log_std = (clipped_log_std - min_log) / (max_log - min_log)

    frame = cv2.cvtColor((normalized_log_std * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    add_text_at_row(frame, f"std, {[min_std, max_std]}, log", 0)
    return frame


def discretized_events_to_cv2_frame(
    colored_event_polarity_sum: Float[np.ndarray, "X Y 3"],
    event_count: Optional[Float[np.ndarray, "B X Y"]] = None,
    output_size: Optional[Tuple[int, int]] = None,
    flip_code: Optional[int] = None,
    illuminance_range: Optional[Tuple[float, float]] = None,
    exponentiation_multiplier: Optional[float] = None,
) -> UInt8[np.ndarray, "X Y 3"]:
    colored_eps = colored_event_polarity_sum
    frame = np.ascontiguousarray((colored_eps * 255).astype(np.uint8))

    if output_size is not None:
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_NEAREST)

    if flip_code is not None:
        frame = cv2.flip(frame, flip_code)

    if event_count is not None:
        count = to_numpy(event_count)
        eppf = count.sum(axis=0).mean()
        add_text_at_row(frame, f"EPPF={eppf:.2f}", 0)
    if illuminance_range is not None:
        add_text_at_row(frame, f"Illum={illuminance_range[0]:.2f}-{illuminance_range[1]:.2f}", 1)
    if exponentiation_multiplier is not None:
        add_text_at_row(frame, f"Exp={exponentiation_multiplier:.2f}", 2)

    return frame
