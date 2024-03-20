from pathlib import Path
from typing import Optional, Tuple

import cv2
import einops
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import nn

from dynamic_fusion.data_generator.configuration import EventDiscretizerConfiguration
from dynamic_fusion.data_generator.event_discretizer import EventDiscretizer
from dynamic_fusion.network_trainer.configuration import SharedConfiguration, TrainerConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.utils.dataset import discretized_events_to_tensors
from dynamic_fusion.utils.datatypes import Events
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.network import stack_and_maybe_unfold_c_list, to_numpy, unfold_temporally
from dynamic_fusion.utils.superresolution import get_spatial_upscaling_output, get_upscaling_pixel_indices_and_distances
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap

CHECKPOINT_DIR = Path("./runs/ready/00_st-un_st-interp_st-up_16")
CHECKPOINT_NAME = "checkpoint_150000"

EVENT_DATA_FILE = Path("./data/raw/e2vid/hdr_selfie.txt")

USE_VIDEO_TO_GENERATE_EVENTS = True
INPUT_VIDEO = Path("./data/raw/e2vid/gnome_huawei_240fps.mp4")
EVS_EXPLORER_CONFIG = "configs/data_generator/simulator/evs_explorer.yml"
DAVIS_CONFIG = "configs/data_generator/simulator/davis_model.yml"

THRESHOLD = 1.35
MIN_ILLUMINANCE = 650
MAX_ILLUMINANCE = 12500

FRAME_SIZE = 0.2  # 200 ms
BINS_PER_FRAME = 2
TAUS_TO_EVALUATE = 5
MAX_HEIGHT = 200
MAX_WIDTH = 300
OUTPUT_VIDEO_PATH = Path("./output.mp4")
OUTPUT_FPS = 10


def main() -> None:
    device = torch.device("cuda")

    config_path = CHECKPOINT_DIR / "config.json"
    with config_path.open("r", encoding="utf8") as f:
        json_config = f.read()
    # Parse the JSON string back into a Configuration instance
    config = TrainerConfiguration.parse_raw(json_config)

    # Load network
    config.network_loader.decoding_checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    config.network_loader.encoding_checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
    encoder, decoder = NetworkLoader(config.network_loader, config.shared).run()
    encoder = encoder.to(device)
    if decoder is None:
        raise NotImplementedError()
    decoder = decoder.to(device)

    if USE_VIDEO_TO_GENERATE_EVENTS:
        events, height, width = get_events_from_video(INPUT_VIDEO, MAX_HEIGHT, MAX_WIDTH)
    else:
        # Load events
        events, height, width = get_events_from_txt(EVENT_DATA_FILE, MAX_HEIGHT, MAX_WIDTH)

    # Discretize events
    max_timestamp = events.timestamp.max() + (FRAME_SIZE - events.timestamp.max() % FRAME_SIZE)
    number_of_temporal_bins = max_timestamp // FRAME_SIZE
    discretizer_config = EventDiscretizerConfiguration(number_of_temporal_bins=number_of_temporal_bins, number_of_temporal_sub_bins_per_bin=BINS_PER_FRAME)
    discretizer = EventDiscretizer(discretizer_config, max_timestamp=max_timestamp)
    discretized_events_dict = discretizer.run({1: events}, (height, width))
    discretized_events = discretized_events_dict[1]

    # Get reconstruction
    reconstruction = run_reconstruction(encoder, decoder, discretized_events, device, config.shared, (height, width), TAUS_TO_EVALUATE)

    # Save video
    size = list(reversed(reconstruction.shape[-2:]))
    size[0] *= 2
    size = tuple(size)  # type: ignore

    colored_event_polarity_sums = img_to_colormap(to_numpy(discretized_events.event_polarity_sum.sum(dim=1)), create_red_blue_cmap(501))

    out = cv2.VideoWriter("output4.mp4", cv2.VideoWriter.fourcc(*"mp4v"), 10, size)
    ms_per_frame = max_timestamp / (TAUS_TO_EVALUATE * number_of_temporal_bins) * 1000

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2
    position = (10, 50)  # Position of the text (bottom left corner)

    for i, frame in enumerate(reconstruction):
        frame = frame[0, :, :, np.newaxis]

        # Flip vertically and rescale to 0-255
        frame_processed = (frame[::-1] * 255).astype(np.uint8)

        if frame_processed.shape[2] == 1:
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)

        i_event_frame = i // TAUS_TO_EVALUATE
        frame_with_events = np.concatenate(((colored_event_polarity_sums[i_event_frame, ::-1] * 255).astype(np.uint8), frame_processed), axis=1)
        cv2.putText(frame_with_events, f"Event frame={i_event_frame}, time={ms_per_frame*i:.0f} ms", position, font, font_scale, font_color, line_type)

        out.write(frame_with_events)

    out.release()


def run_reconstruction(
    encoder: nn.Module,
    decoder: nn.Module,
    discretized_events: DiscretizedEvents,
    device: torch.device,
    config: SharedConfiguration,
    output_shape: Tuple[int, int],
    taus_to_evaluate: int,
) -> Float[torch.Tensor, "T C X Y"]:
    with torch.no_grad():
        event_polarity_sum, timestamp_mean, timestamp_std, event_count = discretized_events_to_tensors(discretized_events)
        number_of_temporal_bins = event_polarity_sum.shape[0]
        eps, means, stds, counts = event_polarity_sum.to(device)[None], timestamp_mean.to(device)[None], timestamp_std.to(device)[None], event_count.to(device)[None]

        encoder.reset_states()
        nearest_pixels, start_to_end_vectors = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), output_shape)

        taus_to_evaluate = 5
        taus = np.arange(0, 1, 1 / taus_to_evaluate)
        taus = torch.tensor(taus).to(device)

        reconstructions = []

        # For unfolding, we need 3, for interpolation also, we need 4
        cs_queue = []

        reconstructions = []
        for t in range(number_of_temporal_bins + 2):
            print(f"{t} / {number_of_temporal_bins + 2}", end="\r")
            reconstructions_t = []
            if t < int(number_of_temporal_bins):
                c_t = encoder(eps[:, t], means[:, t] if config.use_mean else None, stds[:, t] if config.use_std else None, counts[:, t] if config.use_count else None)
                cs_queue.append(stack_and_maybe_unfold_c_list([c_t], config.spatial_unfolding)[0])

                if t == 0:
                    # Makes code easier if we do it this way
                    cs_queue.insert(0, torch.zeros_like(cs_queue[0]))

            if t > 1:
                # We usually have 4 items in the queue, near the end we have 3 or 2
                c_next = None
                if config.temporal_unfolding:
                    c = unfold_temporally(cs_queue, 1)
                    if config.temporal_interpolation and len(cs_queue) > 2:
                        c_next = unfold_temporally(cs_queue, 2)
                else:
                    c = cs_queue[1]  # B X Y C
                    if config.temporal_interpolation and len(cs_queue) > 2:
                        c_next = cs_queue[2]

                for i_tau in range(len(taus)):
                    if config.spatial_upscaling:
                        r_t = get_spatial_upscaling_output(decoder, c, taus[i_tau : i_tau + 1].to(c), c_next, nearest_pixels, start_to_end_vectors)
                    else:
                        tau = einops.repeat(taus[i_tau : i_tau + 1].to(c), "1 -> T X Y 1", X=c.shape[-3], Y=c.shape[-2])
                        r_t = decoder(torch.concat([c, tau], dim=-1))
                        if c_next is not None:
                            r_tnext = decoder(torch.concat([c_next, tau - 1], dim=-1))
                            r_t = r_t * (1 - tau) + r_tnext * (tau)

                    reconstructions_t.append(to_numpy(r_t))

                reconstructions.append(np.stack(reconstructions_t, axis=0))

            if len(cs_queue) > 3:
                del cs_queue[0]

        reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau C X Y
        reconstruction_flat = einops.rearrange(reconstruction_stacked, "tau T C D X Y -> (tau T) (C D) X Y")  # D=1
        return reconstruction_flat

# TODO: just load this as a pd dataframe
def get_events_from_txt(file: Path, max_height: Optional[int] = None, max_width: Optional[int] = None, min_t: Optional[int] = None, max_t: Optional[int] = None) -> Tuple[Events, int, int]:
    lines = file.read_text()
    lines_split = lines.split("\n")

    event_dict = {"timestamp": [], "x": [], "y": [], "polarity": []}

    max_height = max_height if max_height is not None else np.inf  # type: ignore
    max_width = max_width if max_width is not None else np.inf  # type: ignore

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


            if int(y) < max_height and int(x) < max_width:  # type: ignore
                event_dict["timestamp"].append(float(t))
                event_dict["x"].append(int(x))
                event_dict["y"].append(int(y))
                event_dict["polarity"].append(int(p) > 0)

        if len(line.split(" ")) == 2:
            width, height = [int(x) for x in line.split(" ")]

    timestamps = [x - event_dict["timestamp"][0] for x in event_dict["timestamp"]]
    event_dict["timestamp"] = timestamps

    width = width if max_width > width else max_width  # type: ignore
    height = height if max_height > height else max_height  # type: ignore

    return pd.DataFrame(event_dict), height, width


def get_events_from_video(file: Path, max_height: Optional[int] = None, max_width: Optional[int] = None) -> Tuple[Events, int, int]:
    import evs_explorer

    scfg = evs_explorer.Configuration.from_yaml(
        simulator_config=EVS_EXPLORER_CONFIG,
        sensor_config=DAVIS_CONFIG,
        sensor_model="davis_model",
    )

    scfg.input.source = file
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
