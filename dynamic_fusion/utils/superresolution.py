from typing import Tuple

import numpy as np
from jaxtyping import Bool, Float, Int
from numba import jit


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
