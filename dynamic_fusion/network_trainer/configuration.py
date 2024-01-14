from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class SharedConfiguration(BaseModel):
    sequence_length: int = Field(
        ..., description="Length of the sequence used in training."
    )
    resume: bool = Field(
        ...,
        description="If set, resumes training from latest subrun in run directory.",
    )

    use_mean: bool = Field(...)
    use_std: bool = Field(...)
    use_count: bool = Field(...)
    implicit: bool = Field(...)
    feature_unfolding: bool = Field(...)

class AugmentationConfiguration(BaseModel):
    network_image_size: Tuple[int, int] = Field(
        ..., description="Image size that network expects to get."
    )


class DatasetConfiguration(BaseModel):
    dataset_directory: Path = Field(
        ..., description="Path to directory containing the dataset."
    )
    threshold: float = Field(..., description="Threshold to use")

    augmentation_tries: int = Field(
        ...,
        description=(
            "Number of times transforms can be retried before moving onto next image."
        ),
    )

    video_tries: int = Field(
        ..., description="Number of videos to try before raising an exception."
    )

    min_allowed_max_of_mean_polarities_over_times: float = Field(
        ...,
        description=(
            "Minimum allowed value of the maximum of mean polarities taken over times"
            " and thresholds."
        ),
    )

    data_generator_target_image_size: Tuple[int, int] = Field(
        ..., description="Image size that was used in data generation."
    )


class DataHandlerConfiguration(BaseModel):
    augmentation: AugmentationConfiguration = Field(...)  # pyright: ignore
    dataset: DatasetConfiguration = Field(...)  # pyright: ignore
    batch_size: int = Field(..., description="Batch size used in training.")
    num_workers: int = Field(..., description="Workers used by DataLoader")


class EncodingNetworkConfiguration(BaseModel):
    input_size: int = Field(..., description="")
    hidden_size: int = Field(...)
    output_size: int = Field(...)
    kernel_size: int = Field(...)


class DecodingNetworkConfiguration(BaseModel):
    hidden_size: int = Field(...)
    hidden_layers: int = Field(...)


class NetworkLoaderConfiguration(BaseModel):
    encoding: EncodingNetworkConfiguration = Field(...)
    encoding_checkpoint_path: Optional[Path] = Field(...)

    decoding: DecodingNetworkConfiguration = Field(...)
    decoding_checkpoint_path: Optional[Path] = Field(...)


class NetworkFitterConfiguration(BaseModel):
    lr_reconstruction: float = Field(...)
    number_of_training_iterations: int = Field(...)
    reconstruction_loss_name: str = Field(...)
    skip_first_timesteps: int = Field(
        ...,
        description=(
            "Number of initial timesteps to skip when training reconstruction."
        ),
    )

    network_saving_frequency: int = Field(...)
    visualization_frequency: int = Field(...)


class TrainingMonitorConfiguration(BaseModel):
    run_directory: Optional[Path] = Field(...)
    event_colors: List[List[float]] = Field(...)


class TrainerConfiguration(BaseModel):
    shared: SharedConfiguration = Field(...)
    data_handler: DataHandlerConfiguration = Field(...)
    network_loader: NetworkLoaderConfiguration = Field(...)
    network_fitter: NetworkFitterConfiguration = Field(...)
    training_monitor: TrainingMonitorConfiguration = Field(...)
