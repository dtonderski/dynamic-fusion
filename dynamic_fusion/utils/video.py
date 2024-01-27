from typing import List, Literal, Optional, Tuple
import einops

import numpy as np
import torch
from jaxtyping import Float, Shaped
from numpy.random import uniform
from scipy.interpolate import interp1d  # pyright: ignore
from scipy.ndimage import affine_transform  # pyright: ignore
from torchvision.transforms.functional import affine
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import (
    GrayImage,
    GrayVideoFloat,
    GrayVideoInt,
    GrayVideoTorch,
)
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
    target_image_size: Optional[Tuple[int, int]],
    device: torch.device = torch.device("cuda"),
    fill_mode: Literal["wrap", "zeros", "border", "reflection"] = "wrap",
    interpolation: Literal["bilinear", "nearest", "bicubic"] = "bicubic",
) -> GrayVideoTorch:
    """Used to generate images at arbitrary timestamps from an initial
    image and a transform definition. Note that the time of the video
    is assumed to be in [0, 1], where 0 is the timestamp of the first
    frame and 1 is the last frame. Also note that timestamps[0] must be
    0, because when calculating the final transforms, we subtract the
    initial one so that we begin from the start image.

    Args:
        image (GrayImage): _description_
        transform_definition (TransformDefinition): _description_
        timestamps (Float[np.ndarray, " T"]): _description_

    Returns:
        GrayVideoFloat: _description_
    """
    include_first = True
    if timestamps[0] != 0:
        timestamps = np.array([0, *timestamps])
        include_first = False
        # raise ValueError("The first timestamp must always be 0!")
    shifts, rotations, scales = _interpolate_transforms(transform_definition, timestamps)

    transformation_matrices = _transforms_to_matrices(shifts, rotations, scales)
    if not include_first:
        transformation_matrices = transformation_matrices[1:]

    grid = torch.nn.functional.affine_grid(
        torch.tensor(transformation_matrices[:, :2, :]),
        [transformation_matrices.shape[0], 1, *image.shape],
        align_corners=False,
    ).to(device)

    if fill_mode == "wrap":
        # Need to do in-place for memory concerns
        grid.add_(1).remainder_(2).subtract_(1)
        fill_mode = "zeros"

    image_tensor = einops.repeat(
        torch.tensor(image, device=device, dtype=torch.float),
        "H W -> N 1 H W",
        N=transformation_matrices.shape[0],
    )
    video = torch.nn.functional.grid_sample(image_tensor, grid, interpolation, fill_mode, align_corners=False)
    video = normalize(video.squeeze())
    if target_image_size is not None:
        video = crop_video(video, target_image_size)
    return video


def _interpolate_transforms(
    definition: TransformDefinition,
    points_to_interpolate: Float[np.ndarray, "T"],
) -> Tuple[
    Float[np.ndarray, "T 2"],
    Float[np.ndarray, "T 1"],
    Float[np.ndarray, "T 2"],
]:
    shifts = _upsample_knot_values(
        definition.shift_knots,
        points_to_interpolate,
        definition.shift_interpolation,
    )
    shifts -= shifts[0:1, ...]

    rotations = _upsample_knot_values(
        definition.rotation_knots,
        points_to_interpolate,
        definition.rotation_interpolation,
    )
    rotations -= rotations[0:1, ...]

    scales = _upsample_knot_values(
        definition.scale_knots,
        points_to_interpolate,
        definition.scale_interpolation,
    )
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


def _transforms_to_matrices(  # pylint: disable=R0913,R0914
    shifts: Float[np.ndarray, "T 2"],
    rotations: Float[np.ndarray, "T 1"],
    scales: Float[np.ndarray, "T 2"],
) -> Float[np.ndarray, "T 3 3"]:
    transformation_matrices = np.zeros(
        (shifts.shape[0], 3, 3),
        dtype=np.float32,
    )

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
            [[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        scale_matrix = np.array(
            [[1 / scale[0], 0, 0], [0, 1 / scale[1], 0], [0, 0, 1]],
            dtype=np.float32,
        )

        transformation_matrices[step, ...] = scale_matrix @ shift_matrix @ rotation_matrix

    return transformation_matrices


def _generate_video_scipy(image: GrayImage, transformation_matrices: Float[np.ndarray, "T 3 3"]) -> GrayVideoInt:
    video = np.zeros(
        (transformation_matrices.shape[0], image.shape[0], image.shape[1]),
        dtype=np.float32,
    )
    with tqdm(total=len(transformation_matrices), desc="Generating video") as video_progress_bar:
        for timestep, matrix in enumerate(transformation_matrices):
            video[timestep] = affine_transform(image, matrix=matrix)
            video_progress_bar.update(1)

    return video


def _generate_video_torch(
    image: GrayImage,
    angles: List[float],
    translates: List[Tuple[int, int]],
    scales: List[float],
    device: torch.device = torch.device("cuda"),
) -> GrayVideoFloat:
    videos = torch.zeros(len(angles), *image.shape, device=device)
    image_tensor = torch.tensor(image, device=device)[None, ...]

    for i, (angle, translate, scale) in enumerate(zip(angles, translates, scales)):
        videos[i] = affine(
            image_tensor,
            angle=angle,
            translate=translate,
            shear=[0, 0],
            scale=scale,
        )

    return videos.cpu().numpy()


def crop_video(video: GrayVideoFloat, target_image_size: Tuple[int, int]) -> GrayVideoFloat:
    cropped_video_border = (video.shape[-2:] - np.array(target_image_size)) // 2

    cropped_video = video[
        :,
        cropped_video_border[0] : cropped_video_border[0] + target_image_size[0],
        cropped_video_border[1] : cropped_video_border[1] + target_image_size[1],
    ]
    return cropped_video
