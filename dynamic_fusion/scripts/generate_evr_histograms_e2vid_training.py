from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import tqdm

PATH = Path('data/raw/ecoco_depthmaps_test')
OUTPUT_DIR = Path('results/ecoco_depthmaps_histograms')


def main() -> None:
    sequence_dirs = sorted(list(PATH.glob('*')))


    eppf_frames = []
    epps_frames = []

    eppf_sequences = []
    epps_sequences = []

    for sequence_dir in tqdm.tqdm(sequence_dirs):
        voxel_grid_dir = sequence_dir / 'VoxelGrid-betweenframes-5'
        voxel_grid_paths = sorted(list(voxel_grid_dir.glob('*.npy')))
        voxel_grids = [np.load(path) for path in voxel_grid_paths]

        boundary_timestamps = voxel_grid_dir / 'boundary_timestamps.txt'
        timestamps_text = boundary_timestamps.read_text().split('\n')
        voxel_grid_timestamps = {int(x[0]) : (float(x[1]), float(x[2])) for x in [line.split() for line in timestamps_text[:-1]]}

        eppfs = [abs(eps).sum(axis=0).mean() for eps in voxel_grids]
        eppss = [abs(eps).sum(axis=0).mean() / (timestamps[1] - timestamps[0]) for eps, timestamps in zip(voxel_grids, voxel_grid_timestamps.values())]

        eppf_frames.extend(eppfs)
        eppf_sequences.append(sum(eppfs) / len(eppfs))

        epps_frames.extend(eppss)
        epps_sequences.append(sum(eppss) / len(eppss))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(epps_frames, bins=50)
    plt.title('Events per pixel per second, mean over frames')
    plt.xlabel('Events per pixel per second')
    plt.savefig(OUTPUT_DIR / "per_frame.png")

    plt.figure()
    plt.hist(epps_sequences, bins=50)
    plt.title('Events per pixel per second, mean over sequences')
    plt.xlabel('Events per pixel per second')
    plt.savefig(OUTPUT_DIR / "per_sequence.png")

    plt.figure()
    plt.hist(eppf_frames, bins=50)
    plt.title('Events per pixel per frame, mean over frames')
    plt.xlabel('Events per pixel per frame')
    plt.savefig(OUTPUT_DIR / "eppf_per_frame.png")

    plt.figure()
    plt.hist(eppf_sequences, bins=50)
    plt.title('Events per pixel per frame, mean over sequences')
    plt.xlabel('Events per pixel per frame')
    plt.savefig(OUTPUT_DIR / "eppf_per_sequence.png")



if __name__ == '__main__':
    main()
