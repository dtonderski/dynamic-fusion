# mypy: disable-error-code="assignment"
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator


class SharedConfiguration(BaseModel):
    target_image_size: Optional[Tuple[int, int]] = Field(
        None, description="Target image size."
    )
    fps: int = Field(None, description="Framerate of the simulation.")
    number_of_images_to_generate_per_input: int = Field(
        None,
        description="Number of transformed images generated per input image.",
    )
    seed: int = Field(None, description="Randomness seed.")
    overwrite: bool = Field(
        None, description="Controls whether to overwrite existing data."
    )


class ImageLoaderConfiguration(BaseModel):
    dataset_dir: Path = Field(
        None,
        description="Path to the folder containing the images used to generate data.",
    )

    file_extension: str = Field(
        ".jpg", description="Extension of the dataset image files."
    )

    number_of_input_images: int = Field(
        None, description="Number of images to use for data generation."
    )


class ImagePreprocessorConfiguration(BaseModel):
    downscale_probability: float = Field(
        None, description="Probability of downscaling image."
    )

    downscale_range: List[float] = Field(
        None, description="Range of possible scales used for downscaling."
    )
    max_image_size: Optional[Tuple[int, int]]


class VideoGeneratorConfiguration(BaseModel):
    max_number_of_scale_knots: int = Field(
        None, description="Maximum number of scale knots to use."
    )

    max_number_of_shift_knots: int = Field(
        None, description="Maximum number of shift knots to use."
    )

    max_number_of_rotation_knots: int = Field(
        None, description="Maximum number of rotation knots to use."
    )

    max_scale_knot_value: float = Field(
        None, description="Maximum value of scale knots."
    )

    max_shift_knot_multiplier_value: float = Field(
        None, description="Maximum value of shift knot multiplier."
    )

    max_rotation_knot_value: float = Field(
        None, description="Maximum value of rotation knot."
    )

    use_pytorch: bool = Field(
        None, description="Use pytorch for affine transformations."
    )

    pytorch_fill_mode: str = Field(
        ..., description="One of wrap, zeros, border, or reflection"
    )


class EventGeneratorConfiguration(BaseModel):
    sensor_config_path: Path = Field(
        None, description="Path to the sensor config yml."
    )

    simulator_config_path: Path = Field(
        None, description="Path to the simulator config yml."
    )

    thresholds: List[float] = Field(
        None, description="Thresholds to use in data generation."
    )

    min_illuminance_lux_range: List[float] = Field(
        None,
        description="Range of values from which to sample min illuminance.",
    )

    max_illuminance_lux_range: List[float] = Field(
        None,
        description="Range of values from which to sample max illuminance.",
    )


class EventDiscretizerConfiguration(BaseModel):
    number_of_temporal_bins: int = Field(
        None, description="Number of temporal bins to use in discretizer."
    )
    number_of_temporal_sub_bins_per_bin: int = Field(
        None,
        description=(
            "Number of sub-bins to use per temporal bin. If one "
            "discretized event statistic has shape T D H W, then this is the D "
            "dimension."
        ),
    )
    ground_truth_temporal_location_in_bin: str = Field(
        None,
        description=(
            "Location of ground truth image in the bin, must be center or end."
        ),
    )


class DataSaverConfiguration(BaseModel):
    output_dir: Path = Field(
        None,
        description="Path to the folder where to output should be stored.",
    )

    save_events: bool = Field(
        None, description="Determines whether to save the raw events."
    )

    h5_compression: int = Field(None, description="Indicates gzip compression level.")


class DataGeneratorConfiguration(BaseModel):
    shared: SharedConfiguration = SharedConfiguration()  # pyright: ignore

    image_loader: ImageLoaderConfiguration = (
        ImageLoaderConfiguration()
    )  # pyright: ignore

    image_preprocessor: ImagePreprocessorConfiguration = (
        ImagePreprocessorConfiguration()
    )  # pyright: ignore

    video_generator: VideoGeneratorConfiguration = (
        VideoGeneratorConfiguration()
    )  # pyright: ignore

    event_generator: EventGeneratorConfiguration = (
        EventGeneratorConfiguration()
    )  # pyright: ignore

    event_discretizer: EventDiscretizerConfiguration = (
        EventDiscretizerConfiguration()
    )  # pyright: ignore

    data_saver: DataSaverConfiguration = DataSaverConfiguration()  # pyright: ignore

    @root_validator(pre=False)
    @classmethod
    def check_numbers(
        cls, values: Dict[str, BaseModel]
    ) -> Dict[str, BaseModel]:  # pylint: disable
        shared = values.get("shared")
        event_discretizer = values.get("event_discretizer")

        if shared and event_discretizer:
            if (
                shared.number_of_images_to_generate_per_input - 1  # type: ignore
            ) % event_discretizer.number_of_temporal_bins != 0:  # type: ignore
                raise ValueError(
                    "(number_of_images_to_generate_per_input - 1) must be divisible"
                    " by number_of_temporal_bins! For explanation, see docstring of"
                    " function _calculate_indices_of_label_frames in"
                    " event_discretizer.py."
                )

        return values
