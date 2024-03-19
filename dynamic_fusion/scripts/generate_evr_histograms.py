from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
import einops
import torch
from jaxtyping import Float
from tqdm import tqdm

from dynamic_fusion.utils.dataset import CocoTestDataset

TRAIN_DATASET_PATH = Path("data", "interim", "coco", "train", "2subbins")
TEST_DATASET_PATH = Path("data", "interim", "coco", "test", "2subbins")

DATASET_PATHS = [TRAIN_DATASET_PATH, TEST_DATASET_PATH]
DATASET_NAMES = ["train", "test"]

OUTPUT_DIR = Path("results", "evr_histograms")


def get_per_frames_and_per_sequences(event_counts: Float[torch.Tensor, "T B X Y"], bin_size: float) -> Tuple[Float[torch.Tensor, "T"], Float[torch.Tensor, "1"]]:
    event_count_per_frame = einops.reduce(event_counts, "T B X Y -> T X Y", "sum")
    event_count_per_pixel_per_frame = einops.reduce(event_count_per_frame, "T X Y -> T", "mean")
    event_count_per_pixel_per_second = event_count_per_pixel_per_frame / bin_size
    return event_count_per_pixel_per_second, einops.reduce(event_count_per_pixel_per_second, "T -> 1", "mean")


def main() -> None:
    dataset_paths = [TRAIN_DATASET_PATH, TEST_DATASET_PATH]
    dataset_names = ["train", "test"]

    for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
        dataset = CocoTestDataset(dataset_path, threshold=1.3)

        bin_size = 0.2  # CONFIRMED

        event_count_per_pixel_per_second_frames = []
        event_count_per_pixel_per_second_sequences = []

        downscaled_event_count_per_pixel_per_second_frames = []
        downscaled_event_count_per_pixel_per_second_sequences = []

        # pylint: disable=consider-using-enumerate
        for i_sample in tqdm(range(len(dataset))):
            sample = dataset[i_sample]
            discretized_events = sample[7]
            frame_mean, sequence_mean = get_per_frames_and_per_sequences(discretized_events, bin_size)

            event_count_per_pixel_per_second_frames.append(frame_mean)
            event_count_per_pixel_per_second_sequences.append(sequence_mean)

            downscaled_discretized_events = sample[3]
            downscaled_frame_mean, downscaled_sequence_mean = get_per_frames_and_per_sequences(downscaled_discretized_events, bin_size)

            downscaled_event_count_per_pixel_per_second_frames.append(downscaled_frame_mean)
            downscaled_event_count_per_pixel_per_second_sequences.append(downscaled_sequence_mean)


        per_frames = torch.concat(event_count_per_pixel_per_second_frames, dim=0)
        per_sequences = torch.concat(event_count_per_pixel_per_second_sequences, dim=0)

        downscaled_per_frames = torch.concat(downscaled_event_count_per_pixel_per_second_frames, dim=0)
        downscaled_per_sequences = torch.concat(downscaled_event_count_per_pixel_per_second_sequences, dim=0)

        output_dir = OUTPUT_DIR / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)


        plt.figure()
        plt.hist(per_frames, bins=50)
        plt.title('Events per pixel per second, mean over frames')
        plt.xlabel('Events per pixel per second')
        plt.savefig(output_dir / "per_frame.png")

        plt.figure()
        plt.hist(per_sequences, bins=50)
        plt.title('Events per pixel per second, mean over sequences')
        plt.xlabel('Events per pixel per second')
        plt.savefig(output_dir / "per_sequence.png")

        plt.figure()
        plt.hist(downscaled_per_frames, bins=50)
        plt.title('Events per pixel per second, mean over frames (downscaled)')
        plt.xlabel('Events per pixel per second')
        plt.savefig(output_dir / "downscaled_per_frame.png")

        plt.figure()
        plt.hist(downscaled_per_sequences, bins=50)
        plt.title('Events per pixel per second, mean over sequences (downscaled)')
        plt.xlabel('Events per pixel per second')
        plt.savefig(output_dir / "downscaled_per_sequence.png")


if __name__ == "__main__":
    main()
