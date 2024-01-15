from pathlib import Path

import h5py
import numpy as np
import torch

DATASET_DIR = Path("./data/interim/coco/2subbins")
NUMBER_OF_IMAGES_TO_GENERATE_PER_INPUT = 2001
NUMBER_OF_TEMPORAL_BINS = 100
H5_COMPRESSION = 3


if __name__ == '__main__':
    directory_list = [path for path in DATASET_DIR.glob("**/*") if path.is_dir()]
    for directory in directory_list:
        video_path = directory / "ground_truth.h5"
        with h5py.File(video_path, "r") as file:
            video = np.array(file["synchronized_video"])

        input_path = directory / "input.h5"
        with h5py.File(input_path, "r") as file:
            generated_video = np.array(file["generated_video"])

        assert (NUMBER_OF_IMAGES_TO_GENERATE_PER_INPUT - 1) % NUMBER_OF_TEMPORAL_BINS == 0

        discretized_frame_length = (
            NUMBER_OF_IMAGES_TO_GENERATE_PER_INPUT - 1
        ) // NUMBER_OF_TEMPORAL_BINS

        indices = torch.arange(
            discretized_frame_length,
            NUMBER_OF_IMAGES_TO_GENERATE_PER_INPUT,
            discretized_frame_length,
            dtype=torch.int64,
        )
        new_synchronized_video = generated_video[indices]

        with h5py.File(video_path, "w") as file:
            file.create_dataset(
                "/synchronized_video",
                data=new_synchronized_video,
                compression="gzip",
                compression_opts=H5_COMPRESSION,
            )
