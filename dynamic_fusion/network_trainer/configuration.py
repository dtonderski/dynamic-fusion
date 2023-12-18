from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class SharedConfiguration(BaseModel):
    use_mean_and_std: bool = Field(
        ...,
        description="Determines whether mean and std will be used by the network.",
    )
    # TODO: especially add description to this
    control_evaluation_frequency: int = Field(...)
    threshold: float = Field(...)
    batch_size: int = Field(..., description="Batch size used in training.")
    network_image_size: List[int] = Field(
        ..., description="Image size that network expects to get."
    )
    sequence_length: int = Field(
        ..., description="Length of the sequence used in training."
    )
    use_random_control: bool = Field(
        ...,
        description="Use a random control network to train the reconstruction network.",
    )

    @validator("sequence_length")
    @classmethod
    def frequency_must_divide_sequence_length(cls, value, values):  # type: ignore
        if value % values["control_evaluation_frequency"] != 0:
            raise ValueError("Sequence length not divisible!")
        return value


class TransformsConfiguration(BaseModel):
    placeholder: Optional[int] = Field(...)


class DatasetConfiguration(BaseModel):
    dataset_directory: Path = Field(
        ..., description="Path to directory containing the dataset."
    )
    transform_tries: int = Field(
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

class DataHandlerConfiguration(BaseModel):
    transform: TransformsConfiguration = Field(...)  # pyright: ignore
    dataset: DatasetConfiguration = Field(...)  # pyright: ignore


class ReconstructionNetworkConfiguration(BaseModel):
    input_size: int = Field(..., description="")
    hidden_size: int = Field(...)
    output_size: int = Field(...)
    kernel_size: int = Field(...)


class ControlNetworkConfiguration(BaseModel):
    conditioned_control: bool = Field(...)

    kernel_size: int = Field(...)
    number_of_hidden_channels: int = Field(...)
    count_normalizer: float = Field(...)
    sampling_type: str = Field(...)

    @validator("sampling_type")
    @classmethod
    def validate_sampling_type(cls, value: str) -> str:
        allowed_values = ["linear", "prob_interp", "dirichlet_interp"]
        if value in allowed_values:
            return value
        raise ValueError(f"sampling_type must be one of {allowed_values}!")


class NetworkLoaderConfiguration(BaseModel):
    reconstruction: ReconstructionNetworkConfiguration = Field(...)
    reconstruction_checkpoint_path: Optional[str] = Field(...)

    control: ControlNetworkConfiguration = Field(...)
    control_checkpoint_path: Optional[str] = Field(...)


class NetworkFitterConfiguration(BaseModel):
    lr_reconstruction: float = Field(...)
    lr_control: float = Field(...)
    number_of_training_iterations: int = Field(...)
    reconstruction_loss_name: str = Field(...)
    skip_first_timesteps: int = Field(
        ...,
        description=(
            "Number of initial timesteps to skip when training reconstruction."
        ),
    )
    normalize_bandwidth_loss: bool = Field(
        ...,
        description=(
            "Determines whether to normalize the bandwith loss by the first delta."
        ),
    )

    log_temperature: float = Field(...)  # -0.5
    count_loss_weight: float = Field(...)

    network_saving_frequency: int = Field(...)
    soft_mask_mode: str = Field(...)
    hard_mask_mode: str = Field(...)
    min_log_temperature: float = Field(...)

    visualization_frequency: int = Field(...)

    resume: bool = Field(
        ...,
        description="If set, resumes training from latest subrun in run directory.",
    )

    @validator("soft_mask_mode")
    @classmethod
    def validate_soft_mask_mode(cls, value: str) -> str:
        allowed_values = ["softmax", "gumbel", "gumbel_hard"]
        if value in allowed_values:
            return value
        raise ValueError(f"soft_mask_mode must be one of {allowed_values}!")

    @validator("hard_mask_mode")
    @classmethod
    def validate_hard_mask_mode(cls, value: str) -> str:
        allowed_values = ["sample", "gumbel"]
        if value in allowed_values:
            return value
        raise ValueError(f"hard_mask_mode must be one of {allowed_values}!")


class TrainingMonitorConfiguration(BaseModel):
    run_directory: Optional[Path] = Field(...)
    event_colors: List[List[float]] = Field(...)


class TrainerConfiguration(BaseModel):
    shared: SharedConfiguration = Field(...)
    data_handler: DataHandlerConfiguration = Field(...)
    network_loader: NetworkLoaderConfiguration = Field(...)
    network_fitter: NetworkFitterConfiguration = Field(...)
    training_monitor: TrainingMonitorConfiguration = Field(...)

    @validator("network_fitter")
    @classmethod
    def validate_sampling_for_random_control(
        cls, value: NetworkFitterConfiguration, values: dict  # type: ignore
    ) -> NetworkFitterConfiguration:
        if value.hard_mask_mode != "sample" and values["shared"].use_random_control:
            raise ValueError("Hard mask mode must be sample if using random control!")
        return value
