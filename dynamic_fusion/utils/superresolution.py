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
) -> Tuple[Int[np.ndarray, "4 XUpscaled YUpscaled 2"], Float[np.ndarray, "4 XUpscaled YUpscaled 2"], Bool[np.ndarray, "4 XUpscaled YUpscaled"]]:
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

    return nearest_pixels, start_to_end_vectors, out_of_bounds


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
    tau: float,
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
    start_to_end_vectors_normalized = start_to_end_vectors_expanded * einops.repeat(
        torch.tensor(original_resolution), "Dims -> B N X Y Dims", B=b, N=n, X=x, Y=y
    )
    start_to_end_vectors_normalized = start_to_end_vectors_normalized.to(c)

    tau_expanded = einops.repeat(torch.tensor([tau]).to(c), "1 -> B N X Y 1", B=b, N=n, X=x, Y=y)

    r_t = decoding_network(torch.concat([nearest_c, start_to_end_vectors_normalized, tau_expanded], dim=-1))
    if c_next is not None:
        nearest_c_next = c_next[:, nearest_pixels[..., 0], nearest_pixels[..., 1], :]
        r_tnext = decoding_network(torch.concat([nearest_c_next, start_to_end_vectors_normalized, tau_expanded], dim=-1))
        r_t = r_t * (1 - tau_expanded) + r_tnext * (tau_expanded)

    r_t = spatial_bilinear_interpolate(r_t, start_to_end_vectors_normalized)
    r_t = einops.rearrange(r_t, "B X Y C -> B C X Y")

    return r_t
