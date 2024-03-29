from __future__ import annotations

from dataclasses import Field, asdict, dataclass, fields
from typing import Any, Literal, Optional, Tuple

import h5py
import numpy as np
from jaxtyping import Float

GROUP_NAME = "transforms_definition"


@dataclass
class TransformDefinition:
    shift_knots: Float[np.ndarray, "NShiftKnots 2"]
    rotation_knots: Float[np.ndarray, "NRotKnots 1"]
    scale_knots: Float[np.ndarray, "NScaleKnots 2"]
    shift_interpolation: Literal["linear", "cubic"]
    rotation_interpolation: Literal["linear", "cubic"]
    scale_interpolation: Literal["linear", "cubic"]
    target_unscaled_video_size: Optional[Tuple[int, int]]
    down_resolution: Tuple[int, int]

    def save_to_file(self, file: h5py.File) -> None:
        transform_definition_group = file.create_group(GROUP_NAME)

        for key, value in asdict(self).items():
            # No need for compression, it just introduces code complication
            transform_definition_group.create_dataset(
                key,
                data=value,
            )

    @classmethod
    def load_from_file(
        cls,
        file: h5py.File,
    ) -> TransformDefinition:
        def load_field(field_to_load: Field[Any]) -> Any:
            if "np.ndarray" in field_to_load.type:
                return np.array(file[GROUP_NAME][field_to_load.name])
            if "Literal" in field_to_load.type:
                return file[GROUP_NAME][field_to_load.name][()].decode("utf-8")
            if "Optional[Tuple[int, int]]" in field_to_load.type:
                try:
                    return tuple(file[GROUP_NAME][field_to_load.name])
                except Exception:
                    return None
            if "Tuple[int, int]" in field_to_load.type:
                return tuple(int(x) for x in file[GROUP_NAME][field_to_load.name])
            raise ValueError(f"Unhandled type {field_to_load.type}")

        loaded_data = {field.name: load_field(field) for field in fields(cls)}
        return cls(**loaded_data)
