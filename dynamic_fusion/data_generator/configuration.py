# mypy: disable-error-code="assignment"
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator


class SharedConfiguration(BaseModel):
    target_unscaled_image_size: Optional[Tuple[int, int]] = Field(..., description="Target image size used for unscaled events.")
    minimum_downscaled_image_size: Tuple[int, int] = Field(..., description="Minimum allowed downscaled image size used in superresolution.")

    fps: int = Field(..., description="Framerate of the simulation.")
    number_of_images_to_generate_per_input: int = Field(..., description="Number of transformed images generated per input image.")
    seed: int = Field(..., description="Randomness seed.")
    overwrite: bool = Field(..., description="Controls whether to overwrite existing data.")
    downscaling_factor: float = Field(...)


class ImageLoaderConfiguration(BaseModel):
    dataset_dir: Path = Field(..., description="Path to the folder containing the images used to generate data.")
    file_extension: str = Field(".jpg", description="Extension of the dataset image files.")
    number_of_input_images: int = Field(..., description="Number of images to use for data generation.")


class ImagePreprocessorConfiguration(BaseModel):
    max_image_size: Optional[Tuple[int, int]]
    exponentiate: bool = Field(...)
    exponentiation_range: Optional[Tuple[float, float]] = Field(None)


class VideoGeneratorConfiguration(BaseModel):
    max_number_of_scale_knots: int = Field(..., description="Maximum number of scale knots to use.")
    max_number_of_shift_knots: int = Field(..., description="Maximum number of shift knots to use.")

    max_number_of_rotation_knots: int = Field(..., description="Maximum number of rotation knots to use.")
    max_rotation_knot_value: float = Field(..., description="Maximum value of rotation knot.")

    max_scale_knot_value: float = Field(..., description="Maximum value of scale knots.")
    max_shift_knot_multiplier_value: float = Field(..., description="Maximum value of shift knot multiplier.")

    fill_mode: Literal["wrap", "zeros", "border", "reflection"] = Field(..., description="One of wrap, zeros, border, or reflection")


class EventGeneratorConfiguration(BaseModel):
    sensor_config_path: Path = Field(..., description="Path to the sensor config yml.")
    simulator_config_path: Path = Field(..., description="Path to the simulator config yml.")

    thresholds: List[float] = Field(..., description="Thresholds to use in data generation.")

    min_illuminance_lux_range: List[float] = Field(..., description="Range of values from which to sample min illuminance.")
    max_illuminance_lux_range: List[float] = Field(..., description="Range of values from which to sample max illuminance.")


class EventDiscretizerConfiguration(BaseModel):
    number_of_temporal_bins: int = Field(..., description="Number of temporal bins to use in discretizer.")
    number_of_temporal_sub_bins_per_bin: int = Field(
        ..., description="Number of sub-bins to use per temporal bin. If one discretized event statistic has shape T D H W, then this is the D dimension."
    )


class DataSaverConfiguration(BaseModel):
    output_dir: Path = Field(..., description="Path to the folder where to output should be stored.")
    save_events: bool = Field(..., description="Determines whether to save the raw events.")
    save_video: bool = Field(...)
    h5_compression: int = Field(..., description="Indicates gzip compression level.")


class DataGeneratorConfiguration(BaseModel):
    shared: SharedConfiguration = Field(...)
    image_loader: ImageLoaderConfiguration = Field(...)
    image_preprocessor: ImagePreprocessorConfiguration = Field(...)
    video_generator: VideoGeneratorConfiguration = Field(...)
    event_generator: EventGeneratorConfiguration = Field(...)
    event_discretizer: EventDiscretizerConfiguration = Field(...)
    data_saver: DataSaverConfiguration = Field(...)

    @root_validator(pre=False)
    @classmethod
    def check_numbers(cls, values: Dict[str, BaseModel]) -> Dict[str, BaseModel]:  # pylint: disable
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
