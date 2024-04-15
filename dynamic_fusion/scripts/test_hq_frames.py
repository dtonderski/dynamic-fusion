import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
from pandera.typing import DataFrame
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from dynamic_fusion.data_generator.configuration import EventDiscretizerConfiguration
from dynamic_fusion.data_generator.event_discretizer import EventDiscretizer
from dynamic_fusion.network_trainer.configuration import TrainerConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.utils.array import to_numpy
from dynamic_fusion.utils.datatypes import EventSchema
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.evaluation import get_pyiqa_value
from dynamic_fusion.utils.image import scale_to_quantiles
from dynamic_fusion.utils.network import run_reconstruction
from dynamic_fusion.utils.plotting import add_text_at_row, discretized_events_to_cv2_frame, log_std_to_cv2_frame
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap

MODEL = "e2vid_exp"
# MODEL = "e2vid_exp_uncertainty"
MAX_T = 3
SPEED = 0.5

# Data
NAMES = ["boxes", "desk", "desk_fast", "desk_hand_only", "desk_slow"]
# NAMES = ["dynamic_6dof", "boxes_6dof", "calibration", "office_zigzag", "poster_6dof", "shapes_6dof", "slider_depth"]
# NAMES = ["boxes_6dof"]

# Only used in discretization
THRESHOLD = 1.35

DATA_DIR = Path("./data/raw/hq_frames")

OUTPUT_DIR = Path("results/hq_frames/")

# Model
if MODEL == "e2vid_exp":
    CHECKPOINT_DIR = Path("./runs/0323-new-dataset/01_st-un_st-interp_st-up/subrun_00")
elif MODEL == "e2vid_exp_uncertainty":
    CHECKPOINT_DIR = Path("./runs/0323-new-dataset/00_st-un_st-interp_st-up_uncertainty-lpips/subrun_00")

device = torch.device("cuda")

CHECKPOINT_NAME = "latest_checkpoint.pt"


def get_events_from_npy(path: Path) -> DataFrame[EventSchema]:
    p = np.load(path / "events_p.npy")
    t = np.load(path / "events_ts.npy")
    xy = np.load(path / "events_xy.npy")
    x, y = xy.transpose(1, 0)

    return pd.DataFrame({"timestamp": t, "x": x, "y": y, "polarity": p})  # type: ignore


def main() -> None:
    results = {}
    for NAME in NAMES:
        directory = DATA_DIR / NAME

        # -- Load events
        print(NAME, "Loading events")
        events = get_events_from_npy(directory)
        with open(directory / "metadata.json", "r") as json_data:
            metadata = json.load(json_data)
            resolution = metadata["sensor_resolution"]

        frame_ts = np.load(directory / "images_ts.npy")[..., 0]
        if MAX_T is not None:
            events = events[events.timestamp < MAX_T]
        events["frame_bin"] = pd.cut(events.timestamp, frame_ts, labels=False, right=False)
        print(f"Loaded {len(events)} events, with frame_bins ranging from {events.frame_bin.min()} to {events.frame_bin.max()}")

        # -- Discretize events
        print(NAME, "Discretizing events")
        discretizer_config = EventDiscretizerConfiguration(number_of_temporal_bins=1, number_of_temporal_sub_bins_per_bin=2)
        discretizer = EventDiscretizer(discretizer_config, max_timestamp=1.0)

        discretized_frames = []

        for frame_bin, events_in_frame in events.groupby("frame_bin"):
            frame_bin = int(frame_bin)
            timestamp_range = (frame_ts[frame_bin], frame_ts[frame_bin + 1])

            assert np.all((events_in_frame.timestamp < timestamp_range[1]) & (events_in_frame.timestamp >= timestamp_range[0]))
            events_in_frame.timestamp -= timestamp_range[0]
            events_in_frame.timestamp /= timestamp_range[1] - timestamp_range[0]
            discretized_frame = discretizer._discretize_events(events_in_frame, THRESHOLD, resolution)
            discretized_frames.append(discretized_frame)

        # -- Load model
        print(NAME, "Loading model")
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
        decoder = decoder.to(device)  # type: ignore

        # -- Run network
        print(NAME, "Running network")
        discretized_events = DiscretizedEvents.stack_temporally(discretized_frames)
        reconstruction = run_reconstruction(encoder, decoder, discretized_events, device, config.shared)

        reconstruction_norm = reconstruction.copy()
        reconstruction_norm[:, 0] = np.clip(reconstruction_norm[:, 0], 0, 1)

        # -- Calculate metrics
        print(NAME, "Calculating metrics")
        frames = np.load(directory / 'images.npy')[..., 0]
        frames_np = np.stack(frames).astype(np.float32)
        images_quant = scale_to_quantiles(frames_np, [0,1,2], q_low=0.01, q_high=0.99)

        ssim_vals = [ssim(reconstruction_norm[i, 0], images_quant[i], data_range=1) for i in range(len(reconstruction_norm))]

        # plt.plot(ssim_vals)
        # plt.show()

        lpips_vals = []
        lpips = pyiqa.create_metric("lpips", device=device)

        for i in tqdm(range(len(reconstruction_norm))):
            recon_tensor = torch.tensor(reconstruction_norm[i, 0:1][None]).to(device).float()
            image_tensor = torch.tensor(images_quant[i][None, None]).to(device).float()
            lpips_vals.append(get_pyiqa_value(lpips, recon_tensor, image_tensor))

        ssim_val = sum(ssim_vals) / len(ssim_vals)
        lpips_val = sum(lpips_vals) / len(lpips_vals)
        print(NAME, "SSIM:", sum(ssim_vals) / len(ssim_vals))
        print(NAME, "LPIPS:", sum(lpips_vals) / len(lpips_vals))
        results[NAME] = (ssim_val, lpips_val)
        # plt.plot(lpips_vals)
        # plt.show()

        # Create a figure and a set of subplots
        print(NAME, "Plotting")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        size = discretized_events.event_polarity_sum.shape
        out = cv2.VideoWriter(
            f"{str(OUTPUT_DIR)}/{NAME}.mp4",
            cv2.VideoWriter.fourcc(*"mp4v"),
            int(len(discretized_events.event_polarity_sum) / (events.timestamp.max() - events.timestamp.min()) * SPEED),
            (size[-1] * 4, size[-2]),
        )
        colored_event_polarity_sums = img_to_colormap(to_numpy(discretized_events.event_polarity_sum.sum(dim=1)), create_red_blue_cmap(501))
        for I in range(len(reconstruction_norm)):
            frames = []
            event_frame = discretized_events_to_cv2_frame(colored_event_polarity_sums[I], discretized_events.event_count[I])
            recon_frame = cv2.cvtColor((reconstruction_norm[I, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            gt_frame = cv2.cvtColor((images_quant[I] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            add_text_at_row(recon_frame, f"LPIPS={lpips_vals[I]:.2f}", 0)
            add_text_at_row(recon_frame, f"SSIM={ssim_vals[I]:.2f}", 1)

            frames = [event_frame, recon_frame, gt_frame]
            if reconstruction.shape[1] > 1:
                std_frame = log_std_to_cv2_frame(reconstruction_norm[I, 1])
                frames.append(std_frame)

            frame = np.concatenate(frames, axis=1)

            out.write(frame)

        out.release()

    with (OUTPUT_DIR / "result.txt").open("w") as f:
        f.write(str(results))

    for name, (ssim_val, lpips_val) in results.items():
        print(f"{name}, SSIM={ssim_val}, LPIPS={lpips_val}")


if __name__ == "__main__":
    main()
