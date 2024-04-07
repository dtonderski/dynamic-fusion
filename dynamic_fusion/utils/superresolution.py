from typing import Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn


def generate_coords_2d(resolution: Tuple[int, int]) -> Float[np.ndarray, "X Y 2"]:
    a = np.linspace(0, 1, resolution[0])
    b = np.linspace(0, 1, resolution[1])
    A, B = np.meshgrid(a, b)

    # Flatten the arrays and pair them
    return np.array([A, B]).T


def find_corner_pixels(
    start_resolution: Int[np.ndarray, " 2"],
    end_resolution: Tuple[int, int],
    end_coords: Float[np.ndarray, "X Y 2"],
) -> Int[np.ndarray, "4 XUpscaled YUpscaled 2"]:
    corner_pixels = np.zeros((4, *end_resolution, 2), dtype=np.int64)
    two_zeros = np.zeros(2, dtype=np.float64)
    start_resolution = start_resolution.astype(np.float64)

    first_corners = np.floor(end_coords * (start_resolution - 1))
    first_corners = np.clip(first_corners, two_zeros, start_resolution - 2)
    corner_pixels[0] = first_corners
    corner_pixels[1] = first_corners + np.array([1, 0])
    corner_pixels[2] = first_corners + np.array([0, 1])
    corner_pixels[3] = first_corners + np.array([1, 1])

    return corner_pixels


def get_upscaling_pixel_indices_and_distances(
    start_resolution: Tuple[int, int], end_resolution: Tuple[int, int]
) -> Tuple[Int[torch.Tensor, "4 XUpscaled YUpscaled 2"], Float[torch.Tensor, "4 XUpscaled YUpscaled 2"]]:
    start_coords = generate_coords_2d(start_resolution)
    end_coords = generate_coords_2d(end_resolution)

    corner_pixels = find_corner_pixels(np.array(start_resolution), end_resolution, end_coords)

    corner_start_coords = start_coords[corner_pixels[..., 0], corner_pixels[..., 1]]
    corner_to_point_vectors = end_coords[None, :] - corner_start_coords

    return torch.tensor(corner_pixels), torch.tensor(corner_to_point_vectors)


def spatial_bilinear_interpolate(
    r_t: Float[torch.Tensor, "B 4 X Y C"], corner_to_point_vectors_normalized: Float[torch.Tensor, "B 4 X Y 2"]
) -> Float[torch.Tensor, "B X Y C"]:
    distances = 1 - torch.abs(corner_to_point_vectors_normalized)
    weights = distances[..., 0] * distances[..., 1]  # B 4 X Y
    expanded_weights = einops.rearrange(weights, "B N X Y -> B N X Y 1")
    weighted_output = expanded_weights * r_t

    return weighted_output.sum(axis=1)


def get_spatial_upscaling_output(
    decoding_network: nn.Module,
    c: Float[torch.Tensor, "B X Y C"],
    tau: Float[torch.Tensor, " B"],
    c_next: Optional[Float[torch.Tensor, "B X Y C"]],
    corner_pixels: Int[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    corner_to_point_vectors: Float[torch.Tensor, "4 XUpscaled YUpscaled 2"],
) -> Float[torch.Tensor, "B C XUpscaled YUpscaled"]:
    original_resolution = c.shape[-3:-1]

    only_use_first_corner = False
    # Special case when upscaling factor = 1/n, where n is an integer in [1:]. Here, all end points are on a corner, so
    # we only use the first corner, and we can skip computation for the other corners.
    if np.all(corner_to_point_vectors[:,0] == 0):
        only_use_first_corner = True
        corner_pixels = corner_pixels[0:1]
        corner_to_point_vectors = corner_to_point_vectors[0:1]


    corner_pixels = corner_pixels.to(c).long()
    corner_c = c[:, corner_pixels[..., 0], corner_pixels[..., 1], :]  # B 4 X Y C
    b, n, x, y, _ = corner_c.shape

    corner_to_point_vectors_expanded = einops.rearrange(corner_to_point_vectors, "N X Y Dims -> 1 N X Y Dims")
    corner_to_point_vectors_normalized = corner_to_point_vectors_expanded * einops.repeat(torch.tensor(original_resolution) - 1, "Dims -> B N X Y Dims", B=b, N=n, X=x, Y=y)
    corner_to_point_vectors_normalized = corner_to_point_vectors_normalized.to(c)

    tau_expanded = einops.repeat(tau, "B -> B N X Y 1", N=n, X=x, Y=y)


    r_t = decoding_network(torch.concat([corner_c, corner_to_point_vectors_normalized, tau_expanded], dim=-1))
    if c_next is not None:
        corner_c_next = c_next[:, corner_pixels[..., 0], corner_pixels[..., 1], :]
        r_tnext = decoding_network(torch.concat([corner_c_next, corner_to_point_vectors_normalized, tau_expanded - 1], dim=-1))
        r_t = r_t * (1 - tau_expanded) + r_tnext * (tau_expanded)

    if not only_use_first_corner:
        r_t = spatial_bilinear_interpolate(r_t, corner_to_point_vectors_normalized)
    else:
        r_t = r_t[:,0]

    r_t = einops.rearrange(r_t, "B X Y C -> B C X Y")

    return r_t


def get_grid(low_res: Tuple[int, int], high_res: Tuple[int, int], crops_low: Tuple[Tuple[int, int], Tuple[int, int]]) -> Float[torch.Tensor, "1 X Y 2"]:
    # Recommend notebooks/11_grid_testing.ipynb to play around with this
    boundaries = [[crop / low * 2 - 1 for crop in crop_dim] for crop_dim, low in zip(crops_low, low_res)]
    pixel_positions = [torch.linspace(boundary[0], boundary[1], res) for boundary, res in zip(boundaries, high_res)]

    # This combination of ordering and indexing was arrived at partly by trial and error, likely source of bug
    X, Y = torch.meshgrid(*pixel_positions, indexing="ij")
    return torch.stack((Y, X), dim=-1)[None]
