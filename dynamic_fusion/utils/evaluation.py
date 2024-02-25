from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn

from dynamic_fusion.network_trainer.configuration import SharedConfiguration
from dynamic_fusion.utils.dataset import get_ground_truth
from dynamic_fusion.utils.datatypes import CropDefinition, TestBatch
from dynamic_fusion.utils.network import network_test_data_to_device, run_decoder, run_decoder_with_spatial_upsampling, stack_and_maybe_unfold_c_list, to_numpy
from dynamic_fusion.utils.superresolution import get_crop_region, get_upscaling_pixel_indices_and_distances


@torch.no_grad()
def generate_plot(
    batch: TestBatch,
    encoder: nn.Module,
    decoder: nn.Module,
    config: SharedConfiguration,
    device: torch.device,
    upscaling_region: Tuple[int, int] = (0, 0),
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 10,
) -> plt.Figure:
    reconstruction, gt, eps = get_reconstructions_and_gt(batch, encoder, decoder, config, device, upscaling_region, Ts_to_evaluate, taus_to_evaluate)


def get_reconstructions_and_gt(
    batch: TestBatch,
    encoder: nn.Module,
    decoder: nn.Module,
    config: SharedConfiguration,
    device: torch.device,
    inference_region: Optional[Tuple[int, int]] = None,
    Ts_to_evaluate: int = 10,
    taus_to_evaluate: int = 10,
) -> Tuple[Float[torch.Tensor, "T X Y"], Float[torch.Tensor, "T X Y"], Float[torch.Tensor, "T SubBins X Y"]]:
    _, _, _, _, eps_lst, means_lst, stds_lst, counts_lst, images, transforms = network_test_data_to_device(
        batch, device, config.use_mean, config.use_std, config.use_count
    )
    eps, means, stds, counts, image, transform = eps_lst[0], means_lst[0], stds_lst[0], counts_lst[0], images[0], transforms[0]

    c_list = []

    T_max = eps.shape[0]

    needed_Ts = Ts_to_evaluate + config.temporal_unfolding * 2 + config.temporal_interpolation
    t_start = config.temporal_unfolding
    t_end = t_start + Ts_to_evaluate

    T_max_evaluated = (T_max - needed_Ts) // 2 + needed_Ts

    if inference_region is None:
        inference_region = tuple(eps.shape[-2:])

    # Treat each tau as a batch
    taus = einops.repeat(np.arange(0, 1, 1 / taus_to_evaluate), "tau -> tau T", T=needed_Ts)
    crop_definition = CropDefinition(0, 0, T_max_evaluated - needed_Ts + t_start, 0, 0, eps.shape[0], True)
    gt = get_ground_truth(taus[:, t_start:t_end], [image] * taus_to_evaluate, [transform] * taus_to_evaluate, [crop_definition] * taus_to_evaluate, eps.device)
    gt_flat = einops.rearrange(gt, "tau T X Y -> (T tau) X Y")

    # Get regions
    nearest_pixels, start_to_end_vectors, out_of_bounds = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), tuple(gt.shape[-2:]))
    (xmin, xmax, ymin, ymax), upscaled_region = get_crop_region(eps, out_of_bounds, nearest_pixels, inference_region, deterministic=True)

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

    if config.spatial_upsampling:
        cs = torch.zeros(cs_cropped.shape[0], cs_cropped.shape[1], eps.shape[-2], eps.shape[-1], cs_cropped.shape[4])  # type: ignore
        cs = cs.to(cs_cropped)
        cs[:, :, xmin:xmax, ymin:ymax] = cs_cropped
    else:
        cs = cs_cropped

    cs = einops.repeat(cs, "T 1 X Y C -> T tau X Y C", tau=taus_to_evaluate)
    reconstructions = []

    if config.spatial_upsampling:
        taus = einops.rearrange(torch.tensor(taus).to(cs), "tau T -> T tau")
        for _, r_t in run_decoder_with_spatial_upsampling(
            decoder, cs, taus, config.temporal_interpolation, config.temporal_unfolding, nearest_pixels, start_to_end_vectors, t_start, t_end, upscaled_region
        ):
            reconstructions.append(to_numpy(r_t.squeeze()))

    else:
        taus = einops.repeat(torch.tensor(taus).to(cs_cropped), "tau T -> T tau X Y 1", X=cs.shape[-3], Y=cs.shape[-2])  # type: ignore
        for _, r_t in run_decoder(decoder, cs, taus, config.temporal_interpolation, config.temporal_unfolding, t_start, t_end):
            upscaled = [resize(to_numpy(image.squeeze()), output_shape=(xmax_up - xmin_up, ymax_up - ymin_up), order=3, anti_aliasing=True) for image in r_t]
            reconstructions.append(np.stack(upscaled, axis=0))

    reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau X Y
    reconstrucion_flat = einops.rearrange(reconstruction_stacked, "T tau X Y -> (T tau) X Y")

    return reconstrucion_flat, to_numpy(gt_flat[..., xmin_up:xmax_up, ymin_up:ymax_up]), to_numpy(eps[t_start:t_end, ..., xmin:xmax, ymin:ymax])
