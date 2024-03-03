from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class SharedConfiguration(BaseModel):
    sequence_length: int = Field(..., description="Length of the sequence used in training.")
    resume: bool = Field(
        ...,
        description="If set, resumes training from latest subrun in run directory.",
    )

    use_mean: bool = Field(...)
    use_std: bool = Field(...)
    use_count: bool = Field(...)
    implicit: bool = Field(...)
    spatial_unfolding: bool = Field(...)
    temporal_unfolding: bool = Field(...)

    temporal_interpolation: bool = Field(...)
    spatial_upscaling: bool = Field(False)

    min_allowed_max_of_mean_polarities_over_times: float = Field(
        0.05, description="Minimum allowed value of the maximum of mean polarities taken over times and thresholds."
    )


class AugmentationConfiguration(BaseModel):
    network_image_size: Tuple[int, int] = Field(..., description="Size of data that will be fed into the encoder.")


class DatasetConfiguration(BaseModel):
    dataset_directory: Path = Field(..., description="Path to directory containing the dataset.")
    threshold: float = Field(..., description="Threshold to use")

    augmentation_tries: int = Field(..., description="Number of times transforms can be retried before moving onto next image.")
    video_tries: int = Field(..., description="Number of videos to try before raising an exception.")

    max_upscaling: float = Field(...)


class DataHandlerConfiguration(BaseModel):
    augmentation: AugmentationConfiguration = Field(...)
    dataset: DatasetConfiguration = Field(...)
    test_dataset_directory: Path = Field(Path("."))
    test_scale_range: Tuple[int, int] = Field((1, 6))
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
    skip_first_timesteps: int = Field(..., description="Number of initial timesteps to skip when training reconstruction.")
    network_saving_frequency: int = Field(...)
    visualization_frequency: int = Field(...)


class TrainingMonitorConfiguration(BaseModel):
    run_directory: Optional[Path] = Field(...)
    event_colors: List[List[float]] = Field(...)
    persistent_saving_frequency: int = Field(5000)
    Ts_to_visualize: int = Field(50)
    taus_to_visualize: int = Field(3)
    Ts_to_evaluate: int = Field(100)
    taus_to_evaluate: int = Field(5)
    test_samples_to_visualize: List[int] = Field([1,2,3])
    lpips_batch: int = Field(5)


class TrainerConfiguration(BaseModel):
    shared: SharedConfiguration = Field(...)
    data_handler: DataHandlerConfiguration = Field(...)
    network_loader: NetworkLoaderConfiguration = Field(...)
    network_fitter: NetworkFitterConfiguration = Field(...)
    training_monitor: TrainingMonitorConfiguration = Field(...)
    seed: Optional[int] = Field(...)
