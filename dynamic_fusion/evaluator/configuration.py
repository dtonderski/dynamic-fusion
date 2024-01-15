from pathlib import Path
from pydantic import BaseModel, Field


class SharedConfiguration(BaseModel):
    implicit: bool = Field(...)


class DatasetConfiguration(BaseModel):
    dataset_directory: Path = Field(
        ..., description="Path to directory containing the dataset."
    )
    threshold: float = Field(..., description="Threshold to use")


class EvaluatorConfiguration(BaseModel):
    shared: SharedConfiguration = Field(...)
    dataset: DatasetConfiguration = Field(...)
