import itertools
from pathlib import Path

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from dynamic_fusion.network_trainer.configuration import TrainerConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.utils.dataset import CocoTestDataset, collate_test_items
from dynamic_fusion.utils.evaluation import get_reconstructions_and_gt
from dynamic_fusion.utils.loss import LPIPS

LPIPS_BATCH = 200

dataset_path = Path("data", "interim", "coco", "train", "2subbins")
checkpoint_dir = Path("runs/0323-new-dataset/00_st-un_st-interp_st-up_uncertainty-lpips/subrun_00")
TS_TO_EVALUATE = 10
TAUS_TO_EVALUATE = 2
PATCH_SIZE = 32
OUTPUT_DIR = Path("results/std_statistics")


@torch.no_grad()
def main() -> None:
    dataset = CocoTestDataset(dataset_path, (1, 6), threshold=1.35)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / "config.json"
    with config_path.open("r", encoding="utf8") as f:
        json_config = f.read()
    # Parse the JSON string back into a Configuration instance
    config = TrainerConfiguration.parse_raw(json_config)

    config.network_loader.decoding_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
    config.network_loader.encoding_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"

    device = torch.device("cuda")

    encoder, decoder = NetworkLoader(config.network_loader, config.shared).run()
    encoder = encoder.to(device)
    decoder = decoder.to(device)  # type: ignore
    lpips = LPIPS(spatial=False).to(device)

    all_lpips_values = []
    all_correlations = []
    all_median_stds = []
    all_mean_stds = []

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):  # type: ignore
        encoder.reset_states()

        if i > 5:
            break
        batch = collate_test_items([sample])
        recon, gt, _, _ = get_reconstructions_and_gt(
            batch,
            encoder,
            decoder,
            config.shared,
            device,
            scale=1,
            Ts_to_evaluate=TS_TO_EVALUATE,
            taus_to_evaluate=TAUS_TO_EVALUATE,
        )

        # -- Crop and patch
        height, width = recon.shape[2], recon.shape[3]
        crop_height = height % PATCH_SIZE
        crop_width = width % PATCH_SIZE

        # Calculate the starting indices for cropping
        start_crop_height = crop_height // 2
        start_crop_width = crop_width // 2

        # Crop to make the height and width divisible by PATCH_SIZE
        cropped_recon = recon[:, :, start_crop_height : height - (crop_height - start_crop_height), start_crop_width : width - (crop_width - start_crop_width)]
        cropped_gt = gt[:, start_crop_height : height - (crop_height - start_crop_height), start_crop_width : width - (crop_width - start_crop_width)]

        patched_recon = einops.rearrange(cropped_recon, "B C (X PATCH_X) (Y PATCH_Y) -> (B X Y) C PATCH_X PATCH_Y", PATCH_X=PATCH_SIZE, PATCH_Y=PATCH_SIZE)
        patched_gt = einops.rearrange(cropped_gt, "B (X PATCH_X) (Y PATCH_Y) -> (B X Y) PATCH_X PATCH_Y", PATCH_X=PATCH_SIZE, PATCH_Y=PATCH_SIZE)

        correlations = []
        for recon_patch, gt_patch in zip(patched_recon, patched_gt):
            correlations.append(np.corrcoef(recon_patch[0].flatten(), gt_patch.flatten())[0, 1])
        all_correlations.append(np.array(correlations))

        all_median_stds.append(np.median(np.exp(patched_recon[:, 1]), axis=(1, 2)))
        all_mean_stds.append(np.mean(np.exp(patched_recon[:, 1]), axis=(1, 2)))

        lpips_values = []
        for i in range(0, len(patched_recon), LPIPS_BATCH):
            lpips_val = lpips(torch.tensor(patched_recon[i : i + LPIPS_BATCH, 0:1]).cuda(), torch.tensor(patched_gt[i : i + LPIPS_BATCH, None]).cuda())
            lpips_values.append(lpips_val.detach().cpu().numpy().flatten())
        all_lpips_values.append(np.concatenate(lpips_values, axis=0))

    # -- Plotting
    Xs = ["median", "mean"]
    Ys = ["LPIPS", "correlation"]

    lpips_values = np.concatenate(all_lpips_values, axis=0)
    correlations = np.concatenate(all_correlations, axis=0)
    mean_stds = np.concatenate(all_mean_stds, axis=0)
    median_stds = np.concatenate(all_median_stds, axis=0)

    for X, Y in itertools.product(Xs, Ys):

        fig, ax1 = plt.subplots(figsize=(10, 5))

        x = mean_stds if X == "mean" else median_stds
        y = lpips_values if Y == "LPIPS" else correlations

        bin_edges = np.arange(0, 0.25, 0.01)  # Bins from 0 to 1, in steps of 0.05
        bins = np.digitize(x, bin_edges)

        avg_x = [x[bins == i].mean() for i in range(1, len(bin_edges))]
        avg_y = [y[bins == i].mean() for i in range(1, len(bin_edges))]
        counts = [np.sum(bins == i) for i in range(1, len(bin_edges))]

        # Plotting
        ax1.plot(avg_x, avg_y, ".-r", label=Y)

        ax1.set_title(f"{Y} vs {X} STD, patch size = {PATCH_SIZE}, correlation = {np.corrcoef(x, y)[0,1]:.3f}")
        ax1.set_xlabel("Mean STD" if X == "mean" else "Median STD")
        ax1.set_ylabel("LPIPS" if Y == "LPIPS" else "Pearson correlation")
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Patch count")  # we already handled the x-label with ax1
        ax2.plot(avg_x, counts, ".-", label="Patch count")

        fig.legend()

        fig.savefig(f"results/std_statistics/{PATCH_SIZE}_{X}_{Y}.png")


if __name__ == "__main__":
    main()
