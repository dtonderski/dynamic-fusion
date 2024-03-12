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
from dynamic_fusion.utils.loss import LPIPS, UncertaintyLoss
from dynamic_fusion.utils.network import network_test_data_to_device, run_decoder, run_decoder_with_spatial_upscaling, stack_and_maybe_unfold_c_list, to_numpy
from dynamic_fusion.utils.superresolution import get_grid, get_upscaling_pixel_indices_and_distances
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

    @torch.no_grad()
    def reset(self) -> None:
        self.values = []
        self.weights = []


class MetricsDictionary(TypedDict):
    PSNR: Tuple[float, float]
    SSIM: Tuple[float, float]
    MSE: Tuple[float, float]
    LPIPS: Tuple[float, float]
    uncertainty_loss: Optional[Tuple[float, float]]


@torch.no_grad()
def get_reconstructions_and_gt(
    batch: TestBatch,
    encoder: nn.Module,
    decoder: nn.Module,
    config: SharedConfiguration,
    device: torch.device,
    scale: float,
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 10,
) -> Tuple[Float[np.ndarray, "T C X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T SubBins X Y"]]:
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

    output_shape = tuple(int(size * scale) for size in eps_lst[0].shape[-2:])
    input_shape = eps_lst[0].shape[-2:]
    grid = get_grid(input_shape, output_shape, ((0, eps_lst[0].shape[-2]), (0, eps_lst[0].shape[-1]))).to(device)  # type: ignore

    crop_definition = CropDefinition(T_max_evaluated - needed_Ts + t_start, T_max_evaluated - needed_Ts + t_start + t_end, eps.shape[0], grid)
    gt = get_ground_truth(taus[:, t_start:t_end], [image] * taus_to_evaluate, [transform] * taus_to_evaluate, [crop_definition] * taus_to_evaluate, False, eps.device)
    gt_flat = einops.rearrange(gt, "tau T X Y -> (T tau) X Y")

    nearest_pixels, start_to_end_vectors = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), tuple(gt.shape[-2:]))

    for t in range(T_max_evaluated):  # pylint: disable=C0103
        c_t = encoder(
            eps[t][None],
            means[t][None] if config.use_mean else None,
            stds[t][None] if config.use_std else None,
            counts[t][None] if config.use_count else None,
        )
        c_list.append(c_t.clone())

    cs = stack_and_maybe_unfold_c_list(c_list[-needed_Ts:], config.spatial_unfolding)  # Ts_to_evaluate 1 X Y C

    cs = einops.repeat(cs, "T 1 X Y C -> T tau X Y C", tau=taus_to_evaluate)
    reconstructions = []

    for i_tau in range(taus.shape[0]):
        cs_tau, taus_tau = cs[:, i_tau : i_tau + 1], taus[i_tau : i_tau + 1]  # type: ignore
        reconstructions_tau = []
        if config.spatial_upscaling:
            taus_tau = einops.rearrange(torch.tensor(taus_tau).to(cs), "tau T -> T tau")
            for _, r_t in run_decoder_with_spatial_upscaling(
                decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding, nearest_pixels, start_to_end_vectors, t_start, t_end
            ):
                reconstructions_tau.append(to_numpy(r_t))

        else:
            taus_tau = einops.repeat(torch.tensor(taus_tau).to(cs_tau), "tau T -> T tau X Y 1", X=cs.shape[-3], Y=cs.shape[-2])  # type: ignore
            for _, r_t in run_decoder(decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding, t_start, t_end):
                upscaled = [resize(to_numpy(image.squeeze()), output_shape=output_shape, order=3, anti_aliasing=True) for image in r_t]
                reconstructions_tau.append(np.stack(upscaled, axis=0).squeeze())
        reconstructions.append(np.stack(reconstructions_tau, axis=0))

    reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau C X Y
    reconstrucion_flat = einops.rearrange(reconstruction_stacked, "T tau C D X Y -> (tau T) (C D) X Y")  # D=1

    eps_cropped = eps[T_max_evaluated - needed_Ts + t_start : T_max_evaluated - needed_Ts + t_start + t_end]

    gt_cropped = to_numpy(gt_flat)
    gt_downscaled_flat = np.stack([resize(image, eps_cropped.shape[-2:], order=0) for image in gt_cropped], axis=0)

    return (reconstrucion_flat, gt_cropped, gt_downscaled_flat, to_numpy(eps_cropped))


def add_plot_at_t(
    t: int,
    gs: gridspec.GridSpec,
    fig: plt.Figure,
    row: int,
    recon: Float[np.ndarray, "T C X Y"],
    gt: Float[np.ndarray, "T X Y"],
    gt_down: Float[np.ndarray, "T X Y"],
    eps: Float[np.ndarray, "T SubBins X Y"],
    add_title: bool,
    scale: float,
) -> None:
    ax00 = fig.add_subplot(gs[row, 0])
    ax00.imshow(recon[t, 0], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax00.set_title("Reconstruction", fontsize=15)
    ax00.set_xlabel("X", fontsize=15)
    ax00.set_ylabel(f"Y, T={t}", fontsize=15)

    ax01 = fig.add_subplot(gs[row, 1])
    ax01.imshow(gt[t], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax01.set_title("Ground truth", fontsize=15)
    ax01.set_xlabel("X", fontsize=15)

    ax02 = fig.add_subplot(gs[row, 2])
    ax02.imshow(np.abs(gt[t] - recon[t, 0]), cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax02.set_title("|Recon - GT|", fontsize=15)
    ax02.set_xlabel("X", fontsize=15)

    ax03 = fig.add_subplot(gs[row, 3])
    ax03.imshow(gt_down[t], cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax03.set_title("GT downsampled", fontsize=15)
    ax03.set_xlabel("X", fontsize=15)

    T = t * eps.shape[0] // recon.shape[0]
    ax04 = fig.add_subplot(gs[row, 4])
    ax04.imshow(img_to_colormap(eps[T].sum(axis=0), create_red_blue_cmap(501)), cmap="gray", vmin=0, vmax=1, aspect="auto")
    if add_title:
        ax04.set_title(f"Events scale {scale:.2f}", fontsize=15)
    ax04.set_xlabel("X", fontsize=15)

    if recon.shape[1] > 1:
        ax00 = fig.add_subplot(gs[row, 5])
        ax00.imshow(np.exp(recon[t, 1]), cmap="gray", vmin=0, vmax=0.5, aspect="auto")
        if add_title:
            ax00.set_title("STD prediction", fontsize=15)
        ax00.set_xlabel("X", fontsize=15)


def get_evaluation_video(
    recon: Float[np.ndarray, "T C X Y"],
    gt: Float[np.ndarray, "T X Y"],
    gt_down: Float[np.ndarray, "T X Y"],
    eps: Float[np.ndarray, "T SubBins X Y"],
    frames: range,
    scale: float,
) -> FuncAnimation:
    def update(t: int) -> None:
        fig.clf()  # Clear the figure to prepare for the next frame
        # Call your plot function here
        add_plot_at_t(t, gs, fig, 0, recon, gt, gt_down, eps, True, scale)
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.14, wspace=0.14, hspace=0)

    # 19.2 is just 16*6/5, trying to keep figures same size as old
    figsize = (20, 4)
    fig = plt.figure(figsize=figsize, dpi=100)
    plt.close()
    fig.tight_layout()

    columns = 6
    gs = gridspec.GridSpec(1, columns, figure=fig)

    # Create animation
    return FuncAnimation(fig, update, frames=frames, blit=False)  # type: ignore


def get_evaluation_image(
    metrics: MetricsDictionary,
    recon: Float[np.ndarray, "T C X Y"],
    gt: Float[np.ndarray, "T X Y"],
    gt_down: Float[np.ndarray, "T X Y"],
    eps: Float[np.ndarray, "T SubBins X Y"],
    scale: float,
) -> plt.Figure:
    figsize = (16, 6)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    # Define GridSpec layout
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[3, 8, 8])

    table_ax = fig.add_subplot(gs[0, :])
    table_data = [[f"{x[0]:.3f} +- {x[1]:.3f}" for x in metrics.values() if x is not None]]  # type: ignore
    table_col_labels = [key for key, value in metrics.items() if value is not None]

    # Add table to the subplot and remove axis
    colWidths = [0.2] * 5
    table = table_ax.table(cellText=table_data, colLabels=table_col_labels, loc="center", colWidths=colWidths)
    table.auto_set_font_size(False)
    table.scale(1, 1.5)
    table.set_fontsize(15)
    table_ax.axis("off")

    t = recon.shape[0] // 2
    add_plot_at_t(t, gs, fig, 1, recon, gt, gt_down, eps, True, scale)

    y = recon.shape[2] // 2
    ax10 = fig.add_subplot(gs[2, 0])
    ax10.imshow(recon[:, 0, y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax10.set_xlabel("T", fontsize=15)
    ax10.set_ylabel(f"X, Y={y}", fontsize=15)

    ax11 = fig.add_subplot(gs[2, 1])
    ax11.imshow(gt[:, y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax11.set_xlabel("T", fontsize=15)

    ax12 = fig.add_subplot(gs[2, 2])
    ax12.imshow(np.abs(gt[:, y] - recon[:, 0, y]).T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax12.set_xlabel("T", fontsize=15)

    Y = gt_down.shape[1] // 2
    ax13 = fig.add_subplot(gs[2, 3])
    ax13.imshow(gt_down[:, Y].T, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax13.set_xlabel("T", fontsize=15)

    ax14 = fig.add_subplot(gs[2, 4])
    ax14.imshow(img_to_colormap(eps[:, :, Y].sum(axis=1), create_red_blue_cmap(501)).transpose(1, 0, 2), cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax14.set_xlabel("T", fontsize=15)

    if recon.shape[2] > 1:
        ax15 = fig.add_subplot(gs[2, 5])
        ax15.imshow(np.exp(recon[:, 1, y].T), cmap="gray", vmin=0, vmax=0.5, aspect="auto")
        ax15.set_xlabel("T", fontsize=15)

    plt.close()
    return fig


@torch.no_grad()
def get_metrics(
    test_dataset: CocoTestDataset, encoder: nn.Module, decoder: nn.Module, config: SharedConfiguration, device: torch.device, lpips_batch: int = 5
) -> MetricsDictionary:
    psnr, ssim, mse, lpips, uncertainty_loss = PSNR(data_range=1), SSIM(data_range=1), MSE(), LPIPS().to(device), UncertaintyLoss()
    psnrs, ssims, mses, lpipss, uncertainty_losses = [], [], [], [], []

    for i, sample in enumerate(tqdm(test_dataset, total=len(test_dataset))):  # type: ignore
        psnr.reset()
        ssim.reset()
        mse.reset()
        lpips.reset()
        uncertainty_loss.reset()

        batch = collate_test_items([sample])
        scale = test_dataset.scales[i]
        recon, gt, _, _ = get_reconstructions_and_gt(batch, encoder, decoder, config, device, scale=scale, Ts_to_evaluate=10, taus_to_evaluate=5)

        recon = torch.tensor(recon).to(device)
        gt = torch.tensor(gt[:, None]).to(device)

        psnr.update(output=[recon[:, 0:1], gt])
        ssim.update(output=[recon[:, 0:1], gt])
        mse.update(recon[:, 0:1], gt)

        if recon.shape[1] > 1:
            uncertainty_loss.update(recon, gt)

        for i in range(0, len(gt), lpips_batch):
            lpips.update(recon[i : i + lpips_batch, 0:1], gt[i : i + lpips_batch])

        psnrs.append(psnr.compute())
        ssims.append(ssim.compute())
        mses.append(mse.compute())
        lpipss.append(lpips.compute())
        if recon.shape[1] > 1:
            uncertainty_losses.append(uncertainty_loss.compute())

    return {
        "PSNR": (np.mean(psnrs), np.std(psnrs)),
        "SSIM": (np.mean(ssims), np.std(ssims)),
        "MSE": (np.mean(mses), np.std(mses)),  # type: ignore
        "LPIPS": (np.mean(lpipss), np.std(lpipss)),  # type: ignore
        "uncertainty_loss": (np.mean(uncertainty_losses), np.std(uncertainty_losses)) if uncertainty_losses else None,  # type: ignore
    }
