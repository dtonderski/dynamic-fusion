from pathlib import Path

import h5py
from tqdm import tqdm

DATASET_DIR = Path("./data/interim/coco/1subbins")


if __name__ == '__main__':
    directory_list = [path for path in DATASET_DIR.glob("**/*") if path.is_dir()]
    for directory in tqdm(directory_list):
        input_path = directory / "input.h5"
        with h5py.File(input_path, "a") as file:
            if "generated_video" in file.keys():
                del file["generated_video"]
