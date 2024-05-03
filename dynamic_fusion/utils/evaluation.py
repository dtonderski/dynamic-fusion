from typing import List, Optional, Tuple, TypedDict

import einops
import numpy as np
import pyiqa
import torch
from ignite.metrics import PSNR, SSIM
from jaxtyping import Float
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from skimage.transform import resize
from torch import nn
from tqdm import tqdm

from dynamic_fusion.utils.array import to_numpy
from dynamic_fusion.utils.dataset import CocoTestDataset, collate_test_items, get_edge_aps_frames, get_ground_truth, get_initial_aps_frames
from dynamic_fusion.utils.datatypes import CropDefinition, TestBatch
from dynamic_fusion.utils.loss import LPIPS, UncertaintyLoss
from dynamic_fusion.utils.network import network_test_data_to_device, run_decoder, run_decoder_with_spatial_upscaling, stack_and_maybe_unfold_c_list
from dynamic_fusion.utils.superresolution import get_grid, get_upscaling_pixel_indices_and_distances
from dynamic_fusion.utils.visualization import create_red_blue_cmap, img_to_colormap

from .inference_configuration import InferenceConfiguration


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
    LPIPS_ALEX: Tuple[float, float]
    uncertainty_loss: Optional[Tuple[float, float]]
    LSDCI: Optional[Tuple[float, float]]
    LL: Optional[Tuple[float, float]]
    PICP: Optional[Tuple[float, float]]


@torch.no_grad()
def get_reconstructions_and_gt(
    batch: TestBatch,
    encoder: nn.Module,
    decoder: nn.Module,
    config: InferenceConfiguration,
    device: torch.device,
    scale: float,
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 10,
    gt_taus_to_evaluate: Optional[int] = None,
    gt_scale: Optional[float] = None,
) -> Tuple[Float[np.ndarray, "T C X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T X Y"], Float[np.ndarray, "T SubBins X Y"]]:
    encoder.reset_states()

    _, _, _, _, eps_lst, means_lst, stds_lst, counts_lst, images, transforms = network_test_data_to_device(
        batch, device, config.use_mean, config.use_std, config.use_count
    )
    eps, means, stds, counts, image, transform = eps_lst[0], means_lst[0], stds_lst[0], counts_lst[0], images[0], transforms[0]

    c_list = []

    # Treat each tau as a batch
    taus = einops.repeat(np.arange(0, 1, 1 / taus_to_evaluate), "tau -> tau T", T=Ts_to_evaluate)

    recon_output_shape = tuple(int(size * scale) for size in eps_lst[0].shape[-2:])
    gt_output_shape = recon_output_shape if gt_scale is None else tuple(int(size * gt_scale) for size in eps_lst[0].shape[-2:])

    input_shape = eps_lst[0].shape[-2:]
    grid = get_grid(input_shape, gt_output_shape, ((0, eps_lst[0].shape[-2]), (0, eps_lst[0].shape[-1]))).to(device)  # type: ignore

    crop_definition = CropDefinition(0, Ts_to_evaluate, eps.shape[0], grid)
    gt_taus_to_evaluate = taus_to_evaluate if gt_taus_to_evaluate is None else gt_taus_to_evaluate
    gt_taus = einops.repeat(np.arange(0, 1, 1 / gt_taus_to_evaluate), "tau -> tau T", T=Ts_to_evaluate)
    gt = get_ground_truth(
        gt_taus[:, 0:Ts_to_evaluate],
        [image] * gt_taus_to_evaluate,
        [transform] * gt_taus_to_evaluate,
        [crop_definition] * gt_taus_to_evaluate,
        False,
        eps.device,
    )
    gt_flat = einops.rearrange(gt, "tau T X Y -> (T tau) X Y")

    corner_pixels, corner_to_point_vectors = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), recon_output_shape)  # type: ignore
    first_aps_frames = get_initial_aps_frames([image], [transform], [crop_definition], False, device)
    current_frame_info = None

    if config.use_aps_for_all_frames:
        aps_frames = get_edge_aps_frames([image], [transform], [crop_definition], False, device)
    elif config.use_initial_aps_frame:
        first_aps_frames = get_initial_aps_frames([image], [transform], [crop_definition], False, device)

    for t in range(Ts_to_evaluate):  # pylint: disable=C0103
        if config.use_aps_for_all_frames:
            current_frame_info = aps_frames[:, t : t + 2]
        elif config.use_initial_aps_frame:
            current_frame_info = first_aps_frames if t == 0 else torch.zeros_like(first_aps_frames)

        c_t = encoder(
            eps[t][None] if config.use_events else None,
            means[t][None] if config.use_mean and config.use_events else None,
            stds[t][None] if config.use_std and config.use_events else None,
            counts[t][None] if config.use_count and config.use_events else None,
            current_frame_info,
        )
        c_list.append(c_t.clone())

    cs = stack_and_maybe_unfold_c_list(c_list, config.spatial_unfolding)  # Ts_to_evaluate 1 X Y C

    cs = einops.repeat(cs, "T 1 X Y C -> T tau X Y C", tau=taus_to_evaluate)
    reconstructions = []

    for i_tau in range(taus.shape[0]):
        cs_tau, taus_tau = cs[:, i_tau : i_tau + 1], taus[i_tau : i_tau + 1]
        reconstructions_tau = []
        if config.spatial_upscaling:
            taus_tau = einops.rearrange(torch.tensor(taus_tau).to(cs), "tau T -> T tau")
            for _, r_t in run_decoder_with_spatial_upscaling(
                decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding, corner_pixels, corner_to_point_vectors
            ):
                reconstructions_tau.append(to_numpy(r_t))

        else:
            taus_tau = einops.repeat(torch.tensor(taus_tau).to(cs_tau), "tau T -> T tau X Y 1", X=cs.shape[-3], Y=cs.shape[-2])
            for _, r_t in run_decoder(decoder, cs_tau, taus_tau, config.temporal_interpolation, config.temporal_unfolding):
                upscaled = [resize(to_numpy(image.squeeze()), output_shape=recon_output_shape, order=3, anti_aliasing=True) for image in r_t]
                reconstructions_tau.append(np.stack(upscaled, axis=0).squeeze())
        reconstructions.append(np.stack(reconstructions_tau, axis=0))

    reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau C X Y
    reconstrucion_flat = einops.rearrange(reconstruction_stacked, "T tau C D X Y -> (tau T) (C D) X Y")  # D=1

    eps_cropped = eps[:Ts_to_evaluate]

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

    if recon.shape[1] > 1:
        ax15 = fig.add_subplot(gs[2, 5])
        ax15.imshow(np.exp(recon[:, 1, y].T), cmap="gray", vmin=0, vmax=0.5, aspect="auto")
        ax15.set_xlabel("T", fontsize=15)

    plt.close()
    return fig


@torch.no_grad()
def get_metrics(
    test_dataset: CocoTestDataset,
    encoder: nn.Module,
    decoder: nn.Module,
    config: InferenceConfiguration,
    device: torch.device,
    lpips_batch: int = 5,
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 5,
    sequences_to_evaluate: Optional[int] = None,
    gt_taus_to_evaluate: Optional[int] = None,
    gt_scale: Optional[float] = None,
    reduce: bool = True,
    z: float = 1.96,
    patch_size: int = 16,
) -> MetricsDictionary:
    psnr, ssim, mse, lpips, uncertainty_loss = PSNR(data_range=1), SSIM(data_range=1), MSE(), LPIPS(spatial=False).to(device), UncertaintyLoss()
    lpips_alex = pyiqa.create_metric("lpips", device=device)

    # for i in tqdm(range(len(reconstruction_norm))):
    #     recon_tensor = torch.tensor(reconstruction_norm[i, 0:1][None]).to(device).float()
    #     image_tensor = torch.tensor(images[i][None, None]).to(device).float()
    #     lpips_vals.append(get_pyiqa_value(lpips_alex, recon_tensor, image_tensor))

    psnrs, ssims, mses, lpipss, uncertainty_losses, lpipss_alex, lsdcis, lls, picps = [], [], [], [], [], [], [], [], []

    sequences_to_evaluate = sequences_to_evaluate if sequences_to_evaluate is not None else len(test_dataset)

    for i, sample in enumerate(tqdm(test_dataset, total=sequences_to_evaluate)):
        if i >= sequences_to_evaluate:
            break

        psnr.reset()
        ssim.reset()
        mse.reset()
        lpips.reset()
        uncertainty_loss.reset()

        batch = collate_test_items([sample])
        scale = test_dataset.scales[i]
        recon, gt, _, _ = get_reconstructions_and_gt(
            batch,
            encoder,
            decoder,
            config,
            device,
            scale=scale,
            Ts_to_evaluate=Ts_to_evaluate,
            taus_to_evaluate=taus_to_evaluate,
            gt_taus_to_evaluate=gt_taus_to_evaluate,
            gt_scale=gt_scale,
        )

        if gt_taus_to_evaluate:
            # If gt_taus_to_evaluate, then we need to upscale recons so that the shapes match.
            gt_timestamps = np.arange(0, Ts_to_evaluate, 1 / gt_taus_to_evaluate)
            recon_timestamps = np.arange(0, Ts_to_evaluate, 1 / taus_to_evaluate)

            # recon is T C X Y, we only care about first dim
            interpolate = interp1d(recon_timestamps, recon, kind="cubic", axis=0, fill_value="extrapolate")
            recon = interpolate(gt_timestamps).astype(recon.dtype)

        recon = torch.tensor(recon).to(device)
        if gt_scale:
            recon = torch.nn.functional.interpolate(recon, gt.shape[-2:], mode="bicubic", align_corners=True, antialias=True)

        gt = torch.tensor(gt[:, None]).to(device)

        psnr.update(output=[recon[:, 0:1], gt])
        ssim.update(output=[recon[:, 0:1], gt])
        mse.update(recon[:, 0:1], gt)

        if recon.shape[1] > 1:
            uncertainty_loss.update(recon, gt)

        lpipss_alex_sequence = []
        for i in range(0, len(gt), lpips_batch):
            lpips.update(recon[i : i + lpips_batch, 0:1], gt[i : i + lpips_batch])
            # lpips.update(recon[i : i + lpips_batch, 0:1], gt[i : i + lpips_batch])
            lpipss_alex_sequence.append(get_pyiqa_value(lpips_alex, recon[i : i + lpips_batch, 0:1], gt[i : i + lpips_batch]).mean().item())  # type: ignore

        lpipss.append(lpips.compute())
        lpipss_alex.append(np.mean(lpipss_alex_sequence))
        psnrs.append(psnr.compute())
        ssims.append(ssim.compute())
        mses.append(mse.compute())
        if recon.shape[1] > 1:
            lls.append((-np.log(2 * np.pi) / 2 - recon[:, 1] - 1 / 2 * torch.square((gt[:,0] - recon[:, 0]) / torch.exp(recon[:, 1]))).mean().item())
            intervals = [recon[:, 0] - z * torch.exp(recon[:, 1]), recon[:, 0] + z * torch.exp(recon[:, 1])]
            picps.append((torch.logical_and(gt[:,0] > intervals[0], gt[:,0] < intervals[1]).sum() / gt.numel()).item())
            uncertainty_losses.append(uncertainty_loss.compute())

        # -- LSDCI calculation
        height, width = recon.shape[2], recon.shape[3]
        crop_height = height % patch_size
        crop_width = width % patch_size

        # Calculate the starting indices for cropping
        start_crop_height = crop_height // 2
        start_crop_width = crop_width // 2

        # Crop to make the height and width divisible by PATCH_SIZE
        cropped_recon = recon[:, :, start_crop_height : height - (crop_height - start_crop_height), start_crop_width : width - (crop_width - start_crop_width)]
        cropped_gt = gt[:, :, start_crop_height : height - (crop_height - start_crop_height), start_crop_width : width - (crop_width - start_crop_width)]

        patched_recon = einops.rearrange(cropped_recon, "B C (X PATCH_X) (Y PATCH_Y) -> (B X Y) C PATCH_X PATCH_Y", PATCH_X=patch_size, PATCH_Y=patch_size)
        patched_gt = einops.rearrange(cropped_gt, "B C (X PATCH_X) (Y PATCH_Y) -> (B X Y) C PATCH_X PATCH_Y", PATCH_X=patch_size, PATCH_Y=patch_size)

        patch_median_vals = np.median(np.exp(to_numpy(patched_recon[:, 1])), axis=(1, 2))

        patch_lpips_values = []
        for i in range(0, len(patched_recon), lpips_batch):
            lpips_val = lpips(torch.tensor(patched_recon[i : i + lpips_batch, 0:1]).cuda(), torch.tensor(patched_gt[i : i + lpips_batch]).cuda())
            patch_lpips_values.append(to_numpy(lpips_val).flatten())
        patch_lpips_values = np.concatenate(patch_lpips_values, axis=0)

        lsdcis.append(np.corrcoef(patch_median_vals, patch_lpips_values)[0, 1])

    if not reduce:
        return {
            "PSNR": psnrs,  # type: ignore
            "SSIM": ssims,  # type: ignore
            "MSE": mses,  # type: ignore
            "LPIPS": lpipss,  # type: ignore
            "LPIPS_ALEX": lpipss_alex,  # type: ignore
            "uncertainty_loss": uncertainty_losses if uncertainty_losses else None,  # type: ignore
            "LSDCI": lsdcis if lsdcis else None,  # type: ignore
            "LL": lls if lls else None,  # type: ignore
            "PICP": picps if picps else None,  # type: ignore
        }

    return {
        "PSNR": (np.mean(psnrs), np.std(psnrs)),
        "SSIM": (np.mean(ssims), np.std(ssims)),
        "MSE": (np.mean(mses), np.std(mses)),  # type: ignore
        "LPIPS": (np.mean(lpipss), np.std(lpipss)),  # type: ignore
        "LPIPS_ALEX": (np.mean(lpipss_alex), np.std(lpipss_alex)),  # type: ignore
        "uncertainty_loss": (np.mean(uncertainty_losses), np.std(uncertainty_losses)) if uncertainty_losses else None,  # type: ignore
        "LSDCI": (np.mean(lsdcis), np.std(lsdcis)) if lsdcis else None,  # type: ignore
        "LL": (np.mean(lls), np.std(lls)) if lls else None,  # type: ignore
        "PICP": (np.mean(picps), np.std(picps)) if picps else None,  # type: ignore
    }


def get_pyiqa_value(pyiqa_metric: pyiqa, x: Float[torch.Tensor, "B C X Y"], y: Float[torch.Tensor, "B C X Y"]) -> float:
    x = einops.repeat(x, "B C X Y -> B (C three) X Y", three=3)
    y = einops.repeat(y, "B C X Y -> B (C three) X Y", three=3)

    return pyiqa_metric(x, y)  # type: ignore
