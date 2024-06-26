from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize

from dynamic_fusion.data_generator.configuration import EventDiscretizerConfiguration
from dynamic_fusion.data_generator.event_discretizer import EventDiscretizer
from dynamic_fusion.network_trainer.configuration import TrainerConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.utils.array import to_numpy
from dynamic_fusion.utils.datatypes import Events
from dynamic_fusion.utils.network import run_reconstruction
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap

MODEL = "e2vid_exp"
MODEL = "e2vid_exp_uncertainty"

if MODEL == "e2vid_exp":
    CHECKPOINT_DIR = Path("./runs/0323-new-dataset/01_st-un_st-interp_st-up/subrun_00")
elif MODEL == "e2vid_exp_uncertainty":
    CHECKPOINT_DIR = Path("./runs/0323-new-dataset/00_st-un_st-interp_st-up_uncertainty-lpips/subrun_00")

CHECKPOINT_NAME = "latest_checkpoint.pt"

NAME = "selfie"
E2VID_PATH = Path("./data/raw/e2vid")

if NAME == "dynamic":
    EVENT_DATA_FILE = E2VID_PATH / "dynamic_6dof.txt"
elif NAME == "gnome":
    EVENT_DATA_FILE = E2VID_PATH / "gun_bullet_gnome.txt"
elif NAME == "selfie":
    EVENT_DATA_FILE = E2VID_PATH / "hdr_selfie.txt"

MAX_T = 5

USE_VIDEO_TO_GENERATE_EVENTS = False

# Simulated events
if NAME == "dynamic":
    INPUT_VIDEO = None
elif NAME == "gnome":
    INPUT_VIDEO = E2VID_PATH / "gnome_huawei_240fps.mp4"
elif NAME == "selfie":
    INPUT_VIDEO = E2VID_PATH / "selfie_huawei_30fps.mp4"

EVS_EXPLORER_CONFIG = "configs/data_generator/simulator/evs_explorer.yml"
DAVIS_CONFIG = "configs/data_generator/simulator/davis_model.yml"
NUM_FRAMES = 150
THRESHOLD = 1.3
MIN_ILLUMINANCE = 650
MAX_ILLUMINANCE = 12500
DESIRED_WIDTH = 640

# Other settings
FRAME_SIZE = 0.06  # 200 ms
FRAME_SIZE = 0.02  # 200 ms

SCALE = 4
BINS_PER_FRAME = 2
TAUS_TO_EVALUATE = 5

MIN_HEIGHT = None
MAX_HEIGHT = 200

MIN_WIDTH = 550
MAX_WIDTH = None

SPEED = 0.2
OUTPUT_FPS = int(TAUS_TO_EVALUATE / FRAME_SIZE * SPEED)
OUTPUT_VIDEO_PATH = Path(f"./results/upscaled/{MODEL}/{NAME}_simulated={USE_VIDEO_TO_GENERATE_EVENTS}_framesize={FRAME_SIZE}_scale={SCALE}.mp4")

OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    device = torch.device("cuda")

    config_path = CHECKPOINT_DIR / "config.json"
    with config_path.open("r", encoding="utf8") as f:
        json_config = f.read()
    # Parse the JSON string back into a Configuration instance
    config = TrainerConfiguration.parse_raw(json_config)
    print(config)
    # Load network
    config.network_loader.decoding_checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    config.network_loader.encoding_checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    encoder, decoder = NetworkLoader(config.network_loader, config.shared).run()
    encoder = encoder.to(device)
    if decoder is None:
        raise NotImplementedError()
    decoder = decoder.to(device)

    if USE_VIDEO_TO_GENERATE_EVENTS:
        assert INPUT_VIDEO is not None
        events, height, width = get_events_from_video(INPUT_VIDEO, MAX_HEIGHT, MAX_WIDTH, NUM_FRAMES, DESIRED_WIDTH)
    else:
        # Load events
        events, height, width = get_events_from_txt(EVENT_DATA_FILE, MIN_HEIGHT, MAX_HEIGHT, MIN_WIDTH, MAX_WIDTH, max_t=MAX_T)

    print(events, height, width)
    # Discretize events
    max_timestamp = events.timestamp.max() + (FRAME_SIZE - events.timestamp.max() % FRAME_SIZE)
    number_of_temporal_bins = max_timestamp // FRAME_SIZE
    discretizer_config = EventDiscretizerConfiguration(number_of_temporal_bins=number_of_temporal_bins, number_of_temporal_sub_bins_per_bin=BINS_PER_FRAME)
    discretizer = EventDiscretizer(discretizer_config, max_timestamp=max_timestamp)
    discretized_events_dict = discretizer.run({THRESHOLD: events}, (height, width))
    discretized_events = discretized_events_dict[THRESHOLD]
    print(discretized_events.event_polarity_sum.shape)

    out_height, out_width = (int(height * SCALE), int(width * SCALE))

    # Get reconstruction
    reconstruction = run_reconstruction(encoder, decoder, discretized_events, device, config.shared, (out_height, out_width), TAUS_TO_EVALUATE)

    # Save video
    size = list(reversed(reconstruction.shape[-2:]))
    if reconstruction.shape[1] == 1:
        size[0] *= 2
    else:
        size[0] *= 3
    size = tuple(size)  # type: ignore

    colored_event_polarity_sums = img_to_colormap(to_numpy(discretized_events.event_polarity_sum.sum(dim=1)), create_red_blue_cmap(501))
    if SCALE is not None and SCALE != 1:
        resized_eps = [resize(color_eps, (out_height, out_width), order=0) for color_eps in colored_event_polarity_sums]
        colored_event_polarity_sums = np.stack(resized_eps, axis=0)

    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), cv2.VideoWriter.fourcc(*"mp4v"), OUTPUT_FPS, size)
    ms_per_frame = max_timestamp / (TAUS_TO_EVALUATE * number_of_temporal_bins) * 1000

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2
    position = (10, 50)  # Position of the text (bottom left corner)

    for i, frame in enumerate(reconstruction):
        if frame.shape[0] == 2:
            frame = np.concatenate([frame[0, :, :, np.newaxis], np.exp(frame[1, :, :, np.newaxis])], axis=1)
        else:
            frame = frame[0, :, :, np.newaxis]

        frame = np.clip(frame, 0, 1)

        # Flip vertically and rescale to 0-255
        frame_processed = (frame[::-1] * 255).astype(np.uint8)

        if frame_processed.shape[2] == 1:
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)

        i_event_frame = i // TAUS_TO_EVALUATE

        events_per_pixel_per_frame = discretized_events.event_count[i_event_frame].sum(dim=0).mean()

        frame_with_events = np.concatenate(((colored_event_polarity_sums[i_event_frame, ::-1] * 255).astype(np.uint8), frame_processed), axis=1)

        if USE_VIDEO_TO_GENERATE_EVENTS or NAME in ["dynamic"]:
            frame_with_events = np.flip(frame_with_events, 0)
            frame_with_events = np.ascontiguousarray(frame_with_events)

        cv2.putText(
            frame_with_events,
            f"Event frame={i_event_frame}, t={ms_per_frame*i:.0f} ms, EPPF: {events_per_pixel_per_frame: .2f}",
            position,
            font,
            font_scale,
            font_color,
            line_type,
        )

        out.write(frame_with_events)

    out.release()


def get_events_from_txt(
    file: Path,
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    min_t: Optional[float] = None,
    max_t: Optional[float] = None,
    first_row_is_image_shape: bool = True,
    start_at_t_0: bool = True
) -> Tuple[Events, int, int]:

    min_height = min_height if min_height is not None else 0
    min_width = min_width if min_width is not None else 0
    max_height = max_height if max_height is not None else np.inf  # type: ignore
    max_width = max_width if max_width is not None else np.inf  # type: ignore
    min_t = min_t if min_t is not None else 0
    max_t = max_t if max_t is not None else np.inf

    events = pd.read_csv(
        file,
        sep=" ",
        header=None,
        skiprows=[0] if first_row_is_image_shape else [],
        names=["timestamp", "x", "y", "polarity"],
        dtype={"a": np.float64, "x": np.int64, "y": np.int64, "polarity": np.int64},
    )
    if start_at_t_0:
        events.timestamp -= events.timestamp[0]
    events.polarity = events.polarity > 0

    events = events[
        (events.timestamp >= min_t)
        & (events.timestamp < max_t)
        & (events.x >= min_width)
        & (events.x < max_width)
        & (events.y >= min_height)
        & (events.y < max_height)
    ]
    events.x = events.x - min_width
    events.y = events.y - min_height

    if first_row_is_image_shape:
        with open(file, encoding="utf8") as f:
            metadata = f.readline().split(" ")
            assert len(metadata) == 2
            width, height = [int(x) for x in metadata]

        width = min(width, max_width)  # type: ignore
        height = min(height, max_height)  # type: ignore

        width = width - min_width
        height = height - min_height
    else:
        width, height = -1, -1

    return events, height, width


# TODO: just load this as a pd dataframe
def get_events_from_txt_legacy(
    file: Path,
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    min_t: Optional[int] = None,
    max_t: Optional[int] = None,
) -> Tuple[Events, int, int]:
    lines = file.read_text()
    lines_split = lines.split("\n")

    event_dict = {"timestamp": [], "x": [], "y": [], "polarity": []}

    max_height = max_height if max_height is not None else np.inf  # type: ignore
    max_width = max_width if max_width is not None else np.inf  # type: ignore
    min_height = min_height if min_height is not None else 0
    min_width = min_width if min_width is not None else 0
    t_start = None

    for line in lines_split[: len(lines_split)]:
        if len(line.split(" ")) == 4:
            t, x, y, p = line.split(" ")
            if t_start is None:
                t_start = float(t)

            if max_t and float(t) - t_start > max_t:
                break

            if min_t and float(t) - t_start < min_t:
                continue

            if min_height <= int(y) and int(y) < max_height and min_width <= int(x) and int(x) < max_width:  # type: ignore
                event_dict["timestamp"].append(float(t) - t_start)
                event_dict["x"].append(int(x) - min_width)
                event_dict["y"].append(int(y) - min_height)
                event_dict["polarity"].append(int(p) > 0)

        if len(line.split(" ")) == 2:
            width, height = [int(x) for x in line.split(" ")]

    width = min(width, max_width)  # type: ignore
    height = min(height, max_height)  # type: ignore

    width = width - min_width
    height = height - min_height

    return pd.DataFrame(event_dict), height, width


def get_events_from_video(
    file: Path, max_height: Optional[int] = None, max_width: Optional[int] = None, num_frames: Optional[int] = None, desired_width: Optional[int] = None
) -> Tuple[Events, int, int]:
    import evs_explorer

    scfg = evs_explorer.Configuration.from_yaml(
        simulator_config=EVS_EXPLORER_CONFIG,
        sensor_config=DAVIS_CONFIG,
        sensor_model="davis_model",
    )

    scfg.input.source = file
    if num_frames:
        scfg.input.num_frames = num_frames
    if desired_width:
        scfg.input.desired_width = desired_width

    scfg.optics.max_illuminance_lux = MAX_ILLUMINANCE
    scfg.optics.min_illuminance_lux = MIN_ILLUMINANCE
    scfg.sensor.ONth_mul = THRESHOLD
    scfg.sensor.OFFth_mul = THRESHOLD

    evs_explorer = evs_explorer.EvsExplorer(scfg)
    events = evs_explorer.run("sensor_data")

    filtered_events = events
    if max_width:
        filtered_events = filtered_events.loc[filtered_events["x"] < max_width]
    if max_height:
        filtered_events = filtered_events.loc[filtered_events["y"] < max_height]

    height = filtered_events.y.max() + 1
    width = filtered_events.x.max() + 1

    return filtered_events, height, width


if __name__ == "__main__":
    main()
