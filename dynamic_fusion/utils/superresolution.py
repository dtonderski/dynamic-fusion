from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from numba import jit
from torch import nn


def generate_coords_2d(resolution: Tuple[int, int]) -> Float[np.ndarray, "X Y 2"]:
    a = generate_coords_1d(resolution[0])
    b = generate_coords_1d(resolution[1])
    A, B = np.meshgrid(a, b)

    # Flatten the arrays and pair them
    return np.array([A, B]).T


def generate_coords_1d(resolution: int) -> Float[np.ndarray, " N"]:
    return np.linspace(0, 1, resolution + 1)[1:] - 1 / (2 * resolution)


@jit(cache=True)  # type: ignore
def find_nearest_pixels(
    start_resolution: Int[np.ndarray, " 2"],
    end_resolution: Tuple[int, int],
    start_coords: Float[np.ndarray, "X Y 2"],
    end_coords: Float[np.ndarray, "X Y 2"],
) -> Int[np.ndarray, "4 XUpscaled YUpscaled 2"]:
    nearest_pixels = np.zeros((4, *end_resolution, 2))
    for i_end in range(end_resolution[0]):
        for j_end in range(end_resolution[1]):
            end_coord = end_coords[i_end, j_end]
            nearest_pixel = np.floor(end_coord * start_resolution).astype(np.int64)
            vector_from_nearest_start_to_end = end_coord - start_coords[nearest_pixel[0], nearest_pixel[1]]
            next_x, next_y = (vector_from_nearest_start_to_end > 0) * 2 - 1

            nearest_pixels[0, i_end, j_end] = nearest_pixel
            nearest_pixels[1, i_end, j_end] = np.array([nearest_pixel[0] + next_x, nearest_pixel[1]])
            nearest_pixels[2, i_end, j_end] = np.array([nearest_pixel[0], nearest_pixel[1] + next_y])
            nearest_pixels[3, i_end, j_end] = np.array([nearest_pixel[0] + next_x, nearest_pixel[1] + next_y])

    return nearest_pixels


def get_upscaling_pixel_indices_and_distances(
    start_resolution: Tuple[int, int], end_resolution: Tuple[int, int]
) -> Tuple[Int[torch.Tensor, "4 XUpscaled YUpscaled 2"], Float[torch.Tensor, "4 XUpscaled YUpscaled 2"], Bool[torch.Tensor, "4 XUpscaled YUpscaled"]]:
    start_coords = generate_coords_2d(start_resolution)
    end_coords = generate_coords_2d(end_resolution)

    nearest_pixels = find_nearest_pixels(np.array(start_resolution), end_resolution, start_coords, end_coords)

    x_out_of_bounds = np.logical_or(nearest_pixels[..., 0] < 0, nearest_pixels[..., 0] >= start_coords.shape[0])
    y_out_of_bounds = np.logical_or(nearest_pixels[..., 1] < 0, nearest_pixels[..., 1] >= start_coords.shape[1])
    out_of_bounds = np.logical_or(x_out_of_bounds, y_out_of_bounds)
    nearest_pixels[out_of_bounds] = 0

    nearest_pixels = nearest_pixels.astype(int)

    nearest_start_coords = start_coords[nearest_pixels[..., 0], nearest_pixels[..., 1]]
    start_to_end_vectors = end_coords[None, :] - nearest_start_coords

    return torch.tensor(nearest_pixels), torch.tensor(start_to_end_vectors), torch.tensor(out_of_bounds)


def spatial_bilinear_interpolate(
    r_t: Float[torch.Tensor, "B 4 X Y C"], start_to_end_vectors_normalized: Float[torch.Tensor, "B 4 X Y 2"]
) -> Float[torch.Tensor, "B X Y 1"]:
    distances = 1 - torch.abs(start_to_end_vectors_normalized)
    weights = distances[..., 0] * distances[..., 1]  # B 4 X Y
    expanded_weights = einops.rearrange(weights, "B N X Y -> B N X Y 1")
    weighted_output = expanded_weights * r_t

    return weighted_output.sum(axis=1)


def get_spatial_upsampling_output(
    decoding_network: nn.Module,
    c: Float[torch.Tensor, "B X Y C"],
    tau: Float[torch.Tensor, " B"],
    c_next: Optional[Float[torch.Tensor, "B X Y C"]],
    nearest_pixels: Int[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    start_to_end_vectors: Float[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    region_to_upsample: Optional[Tuple[int, int, int, int]] = None,  # Defines x_start, x_stop, y_start, y_stop of region to upsample. Right exclusive
) -> Float[torch.Tensor, "B 1 XUpscaled YUpscaled"]:
    original_resolution = c.shape[-3:-1]

    if region_to_upsample is not None:
        nearest_pixels = nearest_pixels[:, region_to_upsample[0] : region_to_upsample[1], region_to_upsample[2] : region_to_upsample[3]]
        start_to_end_vectors = start_to_end_vectors[:, region_to_upsample[0] : region_to_upsample[1], region_to_upsample[2] : region_to_upsample[3]]

    nearest_pixels = nearest_pixels.to(c).long()
    nearest_c = c[:, nearest_pixels[..., 0], nearest_pixels[..., 1], :]  # B 4 X Y C
    b, n, x, y, _ = nearest_c.shape

    start_to_end_vectors_expanded = einops.rearrange(start_to_end_vectors, "N X Y Dims -> 1 N X Y Dims")
    start_to_end_vectors_normalized = start_to_end_vectors_expanded * einops.repeat(torch.tensor(original_resolution), "Dims -> B N X Y Dims", B=b, N=n, X=x, Y=y)
    start_to_end_vectors_normalized = start_to_end_vectors_normalized.to(c)

    tau_expanded = einops.repeat(tau, "B -> B N X Y 1", N=n, X=x, Y=y)

    r_t = decoding_network(torch.concat([nearest_c, start_to_end_vectors_normalized, tau_expanded], dim=-1))
    if c_next is not None:
        nearest_c_next = c_next[:, nearest_pixels[..., 0], nearest_pixels[..., 1], :]
        r_tnext = decoding_network(torch.concat([nearest_c_next, start_to_end_vectors_normalized, tau_expanded], dim=-1))
        r_t = r_t * (1 - tau_expanded) + r_tnext * (tau_expanded)

    r_t = spatial_bilinear_interpolate(r_t, start_to_end_vectors_normalized)
    r_t = einops.rearrange(r_t, "B X Y C -> B C X Y")

    return r_t


def get_crop_region(
    eps: Float[torch.Tensor, "1 Time SubBin X Y"],
    out_of_bounds: Bool[torch.Tensor, "4 XUpscaled YUpscaled"],
    nearest_pixels: Int[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    upscaling_region_size: Tuple[int, int],
    min_allowed_max_of_mean_polarities_over_times: float = 0,
    deterministic: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    # 1. First, ensure we avoid any pixels whose upsampling required pixels outside of the bounds of the image
    within_bounds_mask = torch.logical_not(out_of_bounds.any(axis=0))

    rows, cols = torch.where(within_bounds_mask)
    xmin_boundary, xmax_boundary = rows.min().item(), (rows.max() - upscaling_region_size[0] + 1).item()
    ymin_boundary, ymax_boundary = cols.min().item(), (cols.max() - upscaling_region_size[1] + 1).item()

    if xmin_boundary > xmax_boundary or ymin_boundary > ymax_boundary:
        raise ValueError("Impossible to find crop region!")

    # 2. Try sampling a crop at most 5 times
    for _ in range(5):
        # 2a. Sample a crop in the upscaled image
        if deterministic:
            xmin_upscaled, ymin_upscaled = int(xmin_boundary), int(ymin_boundary)
        else:
            xmin_upscaled, ymin_upscaled = np.random.randint(low=(xmin_boundary, ymin_boundary), high=(xmax_boundary + 1, ymax_boundary + 1))  # type: ignore
        xmax_upscaled, ymax_upscaled = xmin_upscaled + upscaling_region_size[0], ymin_upscaled + upscaling_region_size[1]

        # 2b. Calculate corresponding region in downscaled image
        used_nearest_pixels = nearest_pixels[:, xmin_upscaled:xmax_upscaled, ymin_upscaled:ymax_upscaled]
        xmin, ymin = used_nearest_pixels.amin(dim=(0, 1, 2))
        xmax, ymax = used_nearest_pixels.amax(dim=(0, 1, 2))

        # 2c. Validate that there's enough events in the downscaled image
        max_of_mean_polarities_over_times = einops.reduce((eps[..., xmin:xmax, ymin:ymax] != 0).float(), "1 Time D X Y -> Time", "mean").max()

        if max_of_mean_polarities_over_times < min_allowed_max_of_mean_polarities_over_times:
            continue
        return ((xmin, xmax, ymin, ymax), (xmin_upscaled, xmax_upscaled, ymin_upscaled, ymax_upscaled))

    # 2d. (optional) If no crop with enough events was found after 5 tries, skip this iteration
    raise ValueError()


def get_grid(low_res: Tuple[int, int], high_res: Tuple[int, int], crops_low: Tuple[Tuple[int, int], Tuple[int, int]]) -> Float[torch.Tensor, "N X Y 2"]:
    # Recommend notebooks/11_grid_testing.ipynb to play around with this
    boundaries = [[crop / low * 2 - 1 for crop in crop_dim] for crop_dim, low in zip(crops_low, low_res)]
    pixel_positions = [torch.linspace(boundary[0], boundary[1], res) for boundary, res in zip(boundaries, high_res)]

    # This combination of ordering and indexing was arrived at partly by trial and error, likely source of bug
    X, Y = torch.meshgrid(*pixel_positions, indexing="ij")
    return torch.stack((Y, X), dim=-1)[None]
