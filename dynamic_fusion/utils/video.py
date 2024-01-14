from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float, Shaped
from numpy.random import uniform
from scipy.interpolate import interp1d  # pyright: ignore
from scipy.ndimage import affine_transform  # pyright: ignore
from torchvision.transforms.functional import affine
from tqdm import tqdm

from dynamic_fusion.utils.datatypes import GrayImage, GrayVideoFloat, GrayVideoInt
from dynamic_fusion.utils.transform import TransformDefinition


def normalize(data: Shaped[np.ndarray, "..."]) -> Shaped[np.ndarray, "..."]:
    data = data - np.min(data)
    return data / np.max(data)


def get_video(
    image: GrayImage,
    transform_definition: TransformDefinition,
    timestamps: Float[np.ndarray, " T"],
    target_image_size: Optional[Tuple[int, int]],
    number_of_images_to_generate_per_input: Optional[int] = None,
    use_pytorch: bool = True,
    device: torch.device = torch.device("cuda"),
) -> GrayVideoFloat:
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
    if timestamps[0] != 0:
        raise ValueError("The first timestamp must always be 0!")
    shifts, rotations, scales = _interpolate_transforms(
        transform_definition, timestamps
    )
    video = _generate_video(
        image,
        shifts,
        rotations,
        scales,
        target_image_size,
        number_of_images_to_generate_per_input,
        use_pytorch,
        device,
    )
    return video


def _generate_video(
    image: GrayImage,
    shifts: Float[np.ndarray, "T 2"],
    rotations: Float[np.ndarray, "T 1"],
    scales: Float[np.ndarray, "T 2"],
    target_image_size: Tuple[int, int],
    number_of_images_to_generate_per_input: Optional[int] = None,
    use_pytorch: bool = True,
    device: torch.device = torch.device("cuda"),
) -> GrayVideoFloat:
    if use_pytorch:
        translates, angles, scales_torch = _transforms_to_torch(
            shifts, rotations, scales
        )
        video = _generate_video_torch(image, angles, translates, scales_torch, device)
    else:
        if number_of_images_to_generate_per_input is None:
            raise ValueError(
                "number_of_images_to_generate_per_input must be set if not using"
                " pytorch!"
            )
        transformation_matrices = _transforms_to_matrices(
            image,
            shifts,
            rotations,
            scales,
            number_of_images_to_generate_per_input,
        )
        video = _generate_video_scipy(image, transformation_matrices)
    video = normalize(video)
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


def _transforms_to_torch(
    shifts: Float[np.ndarray, "T 2"],
    rotations: Float[np.ndarray, "T 1"],
    scales: Float[np.ndarray, "T 2"],
) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
    angles = [float(rotation) * 180 / np.pi for rotation in rotations]
    translates = [tuple(int(x) for x in shift) for shift in shifts]
    torch_scales = [scale[0] for scale in scales]
    return translates, angles, torch_scales  # type: ignore


def _transforms_to_matrices(  # pylint: disable=R0913,R0914
    image: GrayImage,
    shifts: Float[np.ndarray, "T 2"],
    rotations: Float[np.ndarray, "T 1"],
    scales: Float[np.ndarray, "T 2"],
    number_of_images_to_generate_per_input: int,
    rotate_around_center: bool = True,
) -> Float[np.ndarray, "T 3 3"]:
    x_size, y_size = image.shape
    centering_matrix = np.array(
        [[1, 0, x_size / 2.0], [0, 1, y_size / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    inverse_centering_matrix = np.array(
        [[1, 0, -x_size / 2.0], [0, 1, -y_size / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )

    transformation_matrices = np.zeros(
        (number_of_images_to_generate_per_input, 3, 3),
        dtype=np.float32,
    )

    for step in range(number_of_images_to_generate_per_input):
        theta = rotations[step, 0]
        scale = scales[step, :]
        shift = shifts[step, :]

        rotation_matrix = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        shift_matrix = np.array(
            [[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        scale_matrix = np.array(
            [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]],
            dtype=np.float32,
        )

        if not rotate_around_center:
            transformation_matrices[step, ...] = (
                rotation_matrix @ shift_matrix @ scale_matrix
            )
            continue
        transformation_matrices[step, ...] = (
            centering_matrix
            @ rotation_matrix
            @ inverse_centering_matrix
            @ shift_matrix
            @ scale_matrix
        )

    return transformation_matrices


def _generate_video_scipy(
    image: GrayImage, transformation_matrices: Float[np.ndarray, "T 3 3"]
) -> GrayVideoInt:
    video = np.zeros(
        (transformation_matrices.shape[0], image.shape[0], image.shape[1]),
        dtype=np.float32,
    )
    with tqdm(
        total=len(transformation_matrices), desc="Generating video"
    ) as video_progress_bar:
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


def crop_video(
    video: GrayVideoFloat, target_image_size: Tuple[int, int]
) -> GrayVideoFloat:
    cropped_video_border = (video.shape[1:] - np.array(target_image_size)) // 2

    cropped_video = video[
        :,
        cropped_video_border[0] : cropped_video_border[0] + target_image_size[0],
        cropped_video_border[1] : cropped_video_border[1] + target_image_size[1],
    ]
    return cropped_video
