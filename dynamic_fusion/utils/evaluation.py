from typing import List, Optional, Tuple, TypedDict

import einops
import numpy as np
import torch
from ignite.metrics import PSNR, SSIM
from jaxtyping import Float
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.transform import resize
from torch import nn
from tqdm import tqdm

from dynamic_fusion.network_trainer.configuration import SharedConfiguration
from dynamic_fusion.utils.dataset import CocoTestDataset, collate_test_items, get_ground_truth
from dynamic_fusion.utils.datatypes import CropDefinition, TestBatch
from dynamic_fusion.utils.loss import LPIPS
from dynamic_fusion.utils.network import network_test_data_to_device, run_decoder, run_decoder_with_spatial_upscaling, stack_and_maybe_unfold_c_list, to_numpy
from dynamic_fusion.utils.superresolution import get_crop_region, get_upscaling_pixel_indices_and_distances
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap


class MSE:
    values: List[float] = []
    weights: List[int] = []

    @torch.no_grad()
    def update(self, x: Float[torch.Tensor, "B C X Y"], y: Float[torch.Tensor, "B C X Y"]) -> None:
        self.values.append(((x - y) ** 2).mean().item())
        self.weights.append(x.numel())

    @torch.no_grad()
    def compute(self) -> float:
        return sum(value * weight / sum(self.weights) for value, weight in zip(self.values, self.weights))


class MetricsDictionary(TypedDict):
    PSNR: float
    SSIM: float
    MSE: float
    LPIPS: float


@torch.no_grad()
def get_reconstructions_and_gt(
    batch: TestBatch,
    encoder: nn.Module,
    decoder: nn.Module,
    config: SharedConfiguration,
    device: torch.device,
    inference_region: Optional[Tuple[int, int]] = None,
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 10,
) -> Tuple[Float[np.ndarray, "T X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T SubBins X Y"]]:
    encoder.reset_states()

    _, _, _, _, eps_lst, means_lst, stds_lst, counts_lst, images, transforms = network_test_data_to_device(batch, device, config.use_mean, config.use_std, config.use_count)
    eps, means, stds, counts, image, transform = eps_lst[0], means_lst[0], stds_lst[0], counts_lst[0], images[0], transforms[0]

    c_list = []

    T_max = eps.shape[0]

    needed_Ts = Ts_to_evaluate + config.temporal_unfolding * 2 + config.temporal_interpolation
    t_start = config.temporal_unfolding
    t_end = t_start + Ts_to_evaluate

    T_max_evaluated = (T_max - needed_Ts) // 2 + needed_Ts

    # Treat each tau as a batch
    taus = einops.repeat(np.arange(0, 1, 1 / taus_to_evaluate), "tau -> tau T", T=needed_Ts)
    crop_definition = CropDefinition(0, 0, T_max_evaluated - needed_Ts + t_start, 0, 0, eps.shape[0], True)
    gt = get_ground_truth(taus[:, t_start:t_end], [image] * taus_to_evaluate, [transform] * taus_to_evaluate, [crop_definition] * taus_to_evaluate, eps.device)
    gt_flat = einops.rearrange(gt, "tau T X Y -> (T tau) X Y")

    # Get regions
    if inference_region is None:
        inference_region = tuple(x - 8 for x in gt_flat.shape[-2:])  # I think 8 is maximum

    nearest_pixels, start_to_end_vectors, out_of_bounds = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), tuple(gt.shape[-2:]))
    (xmin, xmax, ymin, ymax), upscaled_region = get_crop_region(eps[None], out_of_bounds, nearest_pixels, inference_region, deterministic=True)

    xmin_up, xmax_up, ymin_up, ymax_up = upscaled_region
    for t in range(T_max_evaluated):  # pylint: disable=C0103
        c_t = encoder(
            eps[t][None, ..., xmin:xmax, ymin:ymax],
            means[t][None, ..., xmin:xmax, ymin:ymax] if config.use_mean else None,
            stds[t][None, ..., xmin:xmax, ymin:ymax] if config.use_std else None,
            counts[t][None, ..., xmin:xmax, ymin:ymax] if config.use_count else None,
        )
        c_list.append(c_t.clone())
    cs_cropped = stack_and_maybe_unfold_c_list(c_list[-needed_Ts:], config.spatial_unfolding)  # Ts_to_evaluate 1 X Y C

    if config.spatial_upscaling:
        cs = torch.zeros(cs_cropped.shape[0], cs_cropped.shape[1], eps.shape[-2], eps.shape[-1], cs_cropped.shape[4])  # type: ignore
        cs = cs.to(cs_cropped)
        cs[:, :, xmin:xmax, ymin:ymax] = cs_cropped
    else:
        cs = cs_cropped

    cs = einops.repeat(cs, "T 1 X Y C -> T tau X Y C", tau=taus_to_evaluate)
    reconstructions = []

    for tau in range(taus.shape[0]):
        cs_tau, taus_tau = cs[:, tau : tau + 1], taus[tau : tau + 1]  # type: ignore
        reconstructions_tau = []
        if config.spatial_upscaling:
            taus_tau = einops.rearrange(torch.tensor(taus_tau).to(cs), "tau T -> T tau")
            for _, r_t in run_decoder_with_spatial_upscaling(
                decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding, nearest_pixels, start_to_end_vectors, t_start, t_end, upscaled_region
            ):
                reconstructions_tau.append(to_numpy(r_t.squeeze()))

        else:
            taus_tau = einops.repeat(torch.tensor(taus_tau).to(cs_tau), "tau T -> T tau X Y 1", X=cs.shape[-3], Y=cs.shape[-2])  # type: ignore
            for _, r_t in run_decoder(decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding, t_start, t_end):
                upscaled = [resize(to_numpy(image.squeeze()), output_shape=(xmax_up - xmin_up, ymax_up - ymin_up), order=3, anti_aliasing=True) for image in r_t]
                reconstructions_tau.append(np.stack(upscaled, axis=0).squeeze())
        reconstructions.append(np.stack(reconstructions_tau, axis=0))

    reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau X Y
    reconstrucion_flat = einops.rearrange(reconstruction_stacked, "T tau X Y -> (tau T) X Y")

    eps_cropped = eps[T_max_evaluated - needed_Ts + t_start : T_max_evaluated - needed_Ts + t_start + t_end, ..., xmin:xmax, ymin:ymax]

    gt_cropped = to_numpy(gt_flat[..., xmin_up:xmax_up, ymin_up:ymax_up])
    gt_downscaled_flat = np.stack([resize(image, eps_cropped.shape[-2:], order=0) for image in gt_cropped], axis=0)

    return (reconstrucion_flat, gt_cropped, gt_downscaled_flat, to_numpy(eps_cropped))


def add_plot_at_t(
    t: int,
    gs: gridspec.GridSpec,
    fig: plt.Figure,
    row: int,
    recon: Float[np.ndarray, "T X Y"],
    gt: Float[np.ndarray, "T X Y"],
    gt_down: Float[np.ndarray, "T X Y"],
    eps: Float[np.ndarray, "T SubBins X Y"],
    add_title: bool,
) -> None:
    ax00 = fig.add_subplot(gs[row, 0])
    ax00.imshow(recon[t], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax00.set_title("Reconstruction", fontsize=20)
    ax00.set_xlabel("X", fontsize=20)
    ax00.set_ylabel(f"Y, T={t}", fontsize=20)

    ax01 = fig.add_subplot(gs[row, 1])
    ax01.imshow(gt[t], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax01.set_title("Ground truth", fontsize=20)
    ax01.set_xlabel("X", fontsize=20)

    ax02 = fig.add_subplot(gs[row, 2])
    ax02.imshow(np.abs(gt[t] - recon[t]), cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax02.set_title("|Recon - GT|", fontsize=20)
    ax02.set_xlabel("X", fontsize=20)

    ax03 = fig.add_subplot(gs[row, 3])
    ax03.imshow(gt_down[t], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax03.set_title("GT downsampled", fontsize=20)
    ax03.set_xlabel("X", fontsize=20)

    T = t * eps.shape[0] // recon.shape[0]
    ax04 = fig.add_subplot(gs[row, 4])
    ax04.imshow(img_to_colormap(eps[T].sum(axis=0), create_red_blue_cmap(501)), cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax04.set_title("Events", fontsize=20)
    ax04.set_xlabel("X", fontsize=20)


def get_evaluation_video(
    recon: Float[np.ndarray, "T X Y"], gt: Float[np.ndarray, "T X Y"], gt_down: Float[np.ndarray, "T X Y"], eps: Float[np.ndarray, "T SubBins X Y"], frames: range
) -> FuncAnimation:
    def update(t: int) -> None:
        fig.clf()  # Clear the figure to prepare for the next frame
        # Call your plot function here
        add_plot_at_t(t, gs, fig, 1, recon, gt, gt_down, eps, True)
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.99, top=1.2, bottom=-0.5, wspace=0.14, hspace=0)

    fig = plt.figure(figsize=(16, 4), dpi=100)
    plt.close()
    fig.tight_layout()

    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[6, 15, 0, 15])

    # Create animation
    return FuncAnimation(fig, update, frames=frames, blit=False)  # type: ignore


def get_evaluation_image(
    metrics: MetricsDictionary,
    recon: Float[np.ndarray, "T X Y"],
    gt: Float[np.ndarray, "T X Y"],
    gt_down: Float[np.ndarray, "T X Y"],
    eps: Float[np.ndarray, "T SubBins X Y"],
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 8))
    plt.tight_layout()
    # Define GridSpec layout
    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[6, 15, 0, 15])
    table_ax = fig.add_subplot(gs[0, :])
    table_data = [[f"{value:.4f}" for value in metrics.values()]]
    table_col_labels = list(metrics.keys())

    # Add table to the subplot and remove axis
    table = table_ax.table(cellText=table_data, colLabels=table_col_labels, loc="center", colWidths=[0.2] * 4)
    table.auto_set_font_size(False)
    table.scale(1, 2)
    table.set_fontsize(20)
    table_ax.axis("off")

    t = recon.shape[0] // 2
    add_plot_at_t(t, gs, fig, 1, recon, gt, gt_down, eps, True)

    y = recon.shape[1] // 2
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.imshow(recon[:, y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax10.set_xlabel("T", fontsize=20)
    ax10.set_ylabel(f"X, Y={y}", fontsize=20)

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.imshow(gt[:, y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax11.set_xlabel("T", fontsize=20)

    ax12 = fig.add_subplot(gs[3, 2])
    ax12.imshow(np.abs(gt[:, y] - recon[:, y]).T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax12.set_xlabel("T", fontsize=20)

    Y = gt_down.shape[1] // 2
    ax13 = fig.add_subplot(gs[3, 3])
    ax13.imshow(gt_down[:, Y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax13.set_xlabel("T", fontsize=20)

    ax14 = fig.add_subplot(gs[3, 4])
    ax14.imshow(img_to_colormap(eps[:, :, Y].sum(axis=1), create_red_blue_cmap(501)).transpose(1, 0, 2), cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax14.set_xlabel("T", fontsize=20)
    plt.close()
    return fig


@torch.no_grad()
def get_metrics(test_dataset: CocoTestDataset, encoder: nn.Module, decoder: nn.Module, config: SharedConfiguration, device: torch.device) -> MetricsDictionary:
    psnr, ssim, mse, lpips = PSNR(data_range=1), SSIM(data_range=1), MSE(), LPIPS().to(device)
    for sample in tqdm(test_dataset, total=len(test_dataset)):  # type: ignore
        batch = collate_test_items([sample])
        recon, gt, _, _ = get_reconstructions_and_gt(batch, encoder, decoder, config, device, Ts_to_evaluate=10, taus_to_evaluate=5)

        recon = torch.tensor(recon[:, None]).to(device)
        gt = torch.tensor(gt[:, None]).to(device)

        psnr.update(output=[recon, gt])
        ssim.update(output=[recon, gt])
        mse.update(recon, gt)
        I = 20
        for i in range(0, len(gt), I):
            lpips.update(recon[i : i + I], gt[i : i + I])

    return {"PSNR": psnr.compute(), "SSIM": ssim.compute(), "MSE": mse.compute(), "LPIPS": lpips.compute()}
