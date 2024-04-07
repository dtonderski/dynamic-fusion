from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt

from dynamic_fusion.network_trainer.configuration import TrainerConfiguration
from dynamic_fusion.network_trainer.network_loader import NetworkLoader
from dynamic_fusion.utils.dataset import CocoTestDataset
from dynamic_fusion.utils.evaluation import MetricsDictionary, get_metrics

CHECKPOINT_DIR = Path("./runs/ready/00_st-un_st-interp_st-up_16")
CHECKPOINT_NAME = "checkpoint_150000"
OUTPUT_DIR = Path("results/scale_comparison/")
DATASET_PATH = Path(".", "data", "interim", "coco", "test", "2subbins")
TS_TO_EVALUATE = 10
SEQUENCES_TO_EVALUATE = 3
SPATIAL_SCALES = np.linspace(1, 3, 2)
TEMPORAL_SCALES = range(1, 10, 4)
BOTH_SCALES = range(1, 3)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    def get_metrics_by_scale(temporal_scale: int, spatial_scale: float, max_temporal_scale: int, max_spatial_scale: float) -> MetricsDictionary:
        dataset = CocoTestDataset(DATASET_PATH, (spatial_scale, spatial_scale), threshold=1.3)
        return get_metrics(
            dataset,
            encoder,
            decoder,
            config.shared,
            device,
            Ts_to_evaluate=TS_TO_EVALUATE,
            taus_to_evaluate=temporal_scale,
            sequences_to_evaluate=SEQUENCES_TO_EVALUATE,
            gt_taus_to_evaluate=max_temporal_scale,
            gt_scale=max_spatial_scale,
        )

    max_temporal_scale = TEMPORAL_SCALES[-1]
    max_spatial_scale = SPATIAL_SCALES[-1]

    metrics_spatial_scales = []
    for spatial_scale in SPATIAL_SCALES:
        metrics_spatial_scales.append(get_metrics_by_scale(max_temporal_scale, spatial_scale, max_temporal_scale, max_spatial_scale))

    metrics_temporal_scales = []

    spatial_scale = 1
    for temporal_scale in TEMPORAL_SCALES:
        metrics_temporal_scales.append(get_metrics_by_scale(temporal_scale, max_spatial_scale, max_temporal_scale, max_spatial_scale))

    metrics_both_scales = []
    for scale in BOTH_SCALES:
        metrics_both_scales.append(get_metrics_by_scale(scale, scale, max_temporal_scale, max_spatial_scale))

    fig, ax = plt.subplots()
    ax.plot(SPATIAL_SCALES, [x["LPIPS"][0] for x in metrics_spatial_scales], ".--")
    ax.set_ylabel("LPIPS")
    ax.set_xlabel("spatial_scale")
    ax.set_title("LPIPS, temporal_scale=1")
    ax.grid()
    fig.savefig(OUTPUT_DIR / "spatial_lpips.png")

    fig, ax = plt.subplots()
    ax.plot(SPATIAL_SCALES, [x["SSIM"][0] for x in metrics_spatial_scales], ".--")
    ax.set_ylabel("SSIM")
    ax.set_xlabel("spatial_scale")
    ax.set_title("SSIM, temporal_scale=1")
    ax.grid()
    plt.savefig(OUTPUT_DIR / "spatial_ssim.png")

    fig, ax = plt.subplots()
    ax.plot(TEMPORAL_SCALES, [x["LPIPS"][0] for x in metrics_temporal_scales], ".--")
    ax.set_ylabel("LPIPS")
    ax.set_xlabel("temporal_scale")
    ax.set_title("LPIPS, spatial_scale=1")
    ax.grid()
    fig.savefig(OUTPUT_DIR / "temporal_lpips.png")

    fig, ax = plt.subplots()
    ax.plot(TEMPORAL_SCALES, [x["SSIM"][0] for x in metrics_temporal_scales], ".--")
    ax.set_ylabel("SSIM")
    ax.set_xlabel("temporal_scale")
    ax.set_title("SSIM, spatial_scale=1")
    ax.grid()
    plt.savefig(OUTPUT_DIR / "temporal_ssim.png")

    fig, ax = plt.subplots()
    ax.plot(BOTH_SCALES, [x["LPIPS"][0] for x in metrics_both_scales], ".--")
    ax.set_ylabel("LPIPS")
    ax.set_xlabel("spatiotemporal_scale")
    ax.set_title("LPIPS")
    ax.grid()
    fig.savefig(OUTPUT_DIR / "spatiotemporal_lpips.png")

    fig, ax = plt.subplots()
    ax.plot(BOTH_SCALES, [x["SSIM"][0] for x in metrics_both_scales], ".--")
    ax.set_ylabel("SSIM")
    ax.set_xlabel("spatiotemporal_scale")
    ax.set_title("SSIM")
    ax.grid()
    plt.savefig(OUTPUT_DIR / "spatiotemporal_ssim.png")


if __name__ == "__main__":
    main()
