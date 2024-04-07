from typing import Literal, Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float, Shaped
from numpy.random import uniform
from scipy.interpolate import interp1d  # pyright: ignore

from dynamic_fusion.utils.datatypes import GrayImage, GrayVideoFloat, GrayVideoTorch
from dynamic_fusion.utils.transform import TransformDefinition


def normalize(data: Shaped[np.ndarray, "..."]) -> Shaped[np.ndarray, "..."]:
    if isinstance(data, torch.Tensor):
        data.subtract_(data.min())
        data.divide_(data.max())
        return data

    data = data - data.min()
    return data / data.max()


def get_video(
    image: GrayImage,
    transform_definition: TransformDefinition,
    timestamps: Float[np.ndarray, " T"],
    downscale: bool,
    try_center_crop: bool,
    device: torch.device = torch.device("cuda"),
    fill_mode: Literal["wrap", "zeros", "border", "reflection"] = "wrap",
) -> GrayVideoTorch:
    """Used to generate images at arbitrary timestamps given an initial image and a transform definition. Note that the timestamps have to be in [0, 1].

    Notes:
    1. Grid downscaling happens before cropping. So if image.shape = (512, 512), transform_definition.down_resolution=(256, 128), and
        transform_definition.target_image_size=(96, 96), downscale=True, try_center_crop=True, then the generated video will have shape
        (256,128) and will then be centrally cropped to (96, 96). This is so that no part of the initial image is lost when generating the
        video.

    Args:
        image (GrayImage): Input image.
        transform_definition (TransformDefinition): Definition of the affine transforms.
        timestamps (Float[np.ndarray, " T"]): Timestamps at which to evaluate transform and transform image.
        downscale (bool): if set, the video will be downscaled to transform_definition.down_resolution.
        try_center_crop (bool): if set, the video will be center cropped to transform_definition.target_unscaled_video_size if the attribute is not None.
        device (torch.device, optional): defaults to torch.device("cuda").
        fill_mode (Literal["wrap", "zeros", "border", "reflection"], optional): fill mode to use in grid_sample. Defaults to "wrap".

    Returns:
        GrayVideoTorch: output video.
    """
    # Transform matrices generation
    include_first = True
    if timestamps[0] != 0:
        timestamps = np.array([0, *timestamps])
        include_first = False
        # raise ValueError("The first timestamp must always be 0!")
    shifts, rotations, scales = _interpolate_transforms(transform_definition, timestamps)

    transformation_matrices = _transforms_to_matrices(shifts, rotations, scales)
    if not include_first:
        transformation_matrices = transformation_matrices[1:]
    transformation_matrices = torch.tensor(transformation_matrices[:, :2, :], device=device)

    # Grid creation and manipulation
    T, (X, Y) = transformation_matrices.shape[0], image.shape

    grid = torch.nn.functional.affine_grid(transformation_matrices, [T, 1, X, Y], align_corners=True)
    if downscale:
        # This is equivalent to downscaling the output video, see notebooks/12_interpolation_testing.ipynb
        grid = einops.rearrange(
            torch.nn.functional.interpolate(einops.rearrange(grid, "T X Y C -> T C X Y"), size=transform_definition.down_resolution), "T C X Y -> T X Y C"
        )

    if try_center_crop and transform_definition.target_unscaled_video_size is not None:
        grid = _center_crop_grid(grid, transform_definition.target_unscaled_video_size)

    if fill_mode == "wrap":
        # Need to do in-place for memory concerns
        grid.add_(1).remainder_(2).subtract_(1)
        fill_mode = "zeros"

    # Video generation
    image_tensor = einops.repeat(torch.tensor(image, device=device, dtype=torch.float), "X Y -> T 1 X Y", T=T)
    video = torch.nn.functional.grid_sample(image_tensor, grid, "bicubic", fill_mode, align_corners=True)
    video = normalize(video.squeeze())
    # Make sure shape is T X Y even if only one frame!
    if len(video.shape) == 2:
        video = video[None]
    return video


def _interpolate_transforms(
    definition: TransformDefinition,
    points_to_interpolate: Float[np.ndarray, "T"],
) -> Tuple[
    Float[np.ndarray, "T 2"],
    Float[np.ndarray, "T 1"],
    Float[np.ndarray, "T 2"],
]:
    shifts = _upsample_knot_values(definition.shift_knots, points_to_interpolate, definition.shift_interpolation)
    shifts -= shifts[0:1, ...]

    rotations = _upsample_knot_values(definition.rotation_knots, points_to_interpolate, definition.rotation_interpolation)
    rotations -= rotations[0:1, ...]

    scales = _upsample_knot_values(definition.scale_knots, points_to_interpolate, definition.scale_interpolation)
    scales = scales - scales[0:1, ...] + 1.0
    # print(shifts, rotations, scales)
    return shifts, rotations, scales


def _upsample_knot_values(
    knot_values: Float[np.ndarray, "NKnots Values"],
    points_to_interpolate: Float[np.ndarray, "T"],
    interpolation_type: Literal["random", "linear", "cubic"] = "random",
) -> Float[np.ndarray, "T Values"]:
    r"""This function is used to upsample knot transformation values by
        using interpolation so that they can be used to generate output
        images.

    Args:
        knot_values (Float[np.ndarray, "NKnots Values"]):
            transformation values of the knots.
        interpolation_type (Literal, optional): inteprolation type to
            use. If "random", select randomly between "linear" and
            "cubic". Defaults to "random".

    Returns:
        Float[np.ndarray, "T Values"]: upsampled transformation values
    """

    if interpolation_type == "random":
        interpolation_type = "linear" if uniform() > 0.5 else "cubic"

    number_of_knots = knot_values.shape[0]
    if number_of_knots <= 3:
        interpolation_type = "linear"

    x_knots = np.linspace(0, 1, number_of_knots)
    interpolation = interp1d(x_knots, knot_values, kind=interpolation_type, axis=0)

    return interpolation(points_to_interpolate)


def _transforms_to_matrices(
    shifts: Float[np.ndarray, "T 2"], rotations: Float[np.ndarray, "T 1"], scales: Float[np.ndarray, "T 2"]
) -> Float[np.ndarray, "T 3 3"]:
    transformation_matrices = np.zeros((shifts.shape[0], 3, 3), dtype=np.float32)

    for step, (shift, rotation, scale) in enumerate(zip(shifts, rotations, scales)):
        theta = rotation[0]

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        shift_matrix = np.array(
            [
                [1, 0, -shift[0]],
                [0, 1, -shift[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        scale_matrix = np.array(
            [
                [1 / scale[0], 0, 0],
                [0, 1 / scale[1], 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        transformation_matrices[step, ...] = scale_matrix @ shift_matrix @ rotation_matrix

    return transformation_matrices


def _center_crop_grid(grid: Float[torch.Tensor, "T X Y 2"], target_image_size: Tuple[int, int]) -> GrayVideoFloat:
    _, X, Y, _ = grid.shape
    x_min, y_min = (X - target_image_size[0]) // 2, (Y - target_image_size[1]) // 2

    cropped_video = grid[
        :,
        x_min : x_min + target_image_size[0],
        y_min : y_min + target_image_size[1],
    ]
    return cropped_video
