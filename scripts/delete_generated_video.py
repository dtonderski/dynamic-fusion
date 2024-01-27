import os
from pathlib import Path

import h5py
from tqdm import tqdm

DATASET_DIR = Path("./data/interim/coco/2subbins_new")

if __name__ == "__main__":
    directory_list = [path for path in DATASET_DIR.glob("**/*") if path.is_dir()]
    for directory in tqdm(directory_list):
        input_path = directory / "input.h5"
        temp_path = input_path.with_name("temp.h5")
        if temp_path.exists():
            os.remove(temp_path)
        with h5py.File(input_path, "a") as file_src, h5py.File(input_path.with_name("temp.h5"), "a") as file_dst:
            for a in file_src.attrs:
                file_dst.attrs[a] = file_src.attrs[a]
            for dataset in file_src:
                if not "generated_video" in dataset:
                    file_src.copy(dataset, file_dst)

        os.rename(temp_path, input_path)
