from pydantic import BaseModel, Field


class InferenceConfiguration(BaseModel):
    use_events: bool = Field(True)
    use_mean: bool = Field(...)
    use_std: bool = Field(...)
    use_count: bool = Field(...)
    spatial_unfolding: bool = Field(...)
    temporal_unfolding: bool = Field(...)

    temporal_interpolation: bool = Field(...)
    spatial_upscaling: bool = Field(False)

    use_aps_for_all_frames: bool = Field(False)
    use_initial_aps_frame: bool = Field(False)
