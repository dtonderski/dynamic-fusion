from typing import Generator, List, Optional, Tuple, Union
import einops

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import nn

from dynamic_fusion.utils.datatypes import Batch, TestBatch
from dynamic_fusion.utils.superresolution import get_spatial_upscaling_output


def network_data_to_device(
    batch: Batch,
    device: torch.device,
    use_mean: bool,
    use_std: bool,
    use_count: bool,
) -> Batch:
    (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, preprocessed_image, transforms_definition, crops_definition) = batch

    event_polarity_sums = event_polarity_sums.to(device)
    if use_mean:
        timestamp_means = timestamp_means.to(device)
    if use_std:
        timestamp_stds = timestamp_stds.to(device)
    if use_count:
        event_counts = event_counts.to(device)

    for i, crop in enumerate(crops_definition):
        crops_definition[i].grid = crop.grid.to(device)

    return (event_polarity_sums, timestamp_means, timestamp_stds, event_counts, preprocessed_image, transforms_definition, crops_definition)


def network_test_data_to_device(
    batch: TestBatch,
    device: torch.device,
    use_mean: bool,
    use_std: bool,
    use_count: bool,
) -> TestBatch:

    eps, means, stds, counts, down_eps, down_means, down_stds, down_counts, preprocessed_image, transform = batch
    eps = eps.to(device)
    down_eps = [x.to(device) for x in down_eps]

    if use_mean:
        means = means.to(device)
        down_means = [x.to(device) for x in down_means]
    if use_std:
        stds = stds.to(device)
        down_stds = [x.to(device) for x in down_stds]
    if use_count:
        counts = counts.to(device)
        down_counts = [x.to(device) for x in down_counts]

    return (eps, means, stds, counts, down_eps, down_means, down_stds, down_counts, preprocessed_image, transform)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:  # type: ignore[type-arg]
    return tensor.detach().cpu().numpy()  # type: ignore[no-any-return]


def run_decoder(
    decoder: nn.Module,
    cs: Float[torch.Tensor, "T B X Y C"],
    taus: Float[torch.Tensor, "T B X Y 1"],
    temporal_interpolation: bool,
    temporal_unfolding: bool,
    t_start: int = 0,
) -> Generator[Tuple[int, Float[torch.Tensor, "B X Y Cout"]], None, None]:
    for t in range(t_start, cs.shape[0]):
        c_next = None
        if temporal_unfolding:
            c = unfold_temporally(cs, t)
            if temporal_interpolation and t < cs.shape[0]:
                c_next = unfold_temporally(cs, t + 1)
        else:
            c = cs[t]  # B X Y C
            if temporal_interpolation and t < cs.shape[0]:
                c_next = cs[t + 1]

        r_t = decoder(torch.concat([c, taus[t]], dim=-1))
        if c_next is not None:
            r_tnext = decoder(torch.concat([c_next, taus[t] - 1], dim=-1))
            r_t = r_t * (1 - taus[t]) + r_tnext * (taus[t])
        yield t, r_t
    return


def run_decoder_with_spatial_upscaling(
    decoder: nn.Module,
    cs: Float[torch.Tensor, "T B X Y C"],
    taus: Float[torch.Tensor, "T B"],
    temporal_interpolation: bool,
    temporal_unfolding: bool,
    corner_pixels: Int[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    corner_to_point_vectors: Float[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    t_start: int = 0,
) -> Generator[Tuple[int, Float[torch.Tensor, "B X Y Cout"]], None, None]:
    for t in range(t_start, cs.shape[0]):
        c_next = None
        if temporal_unfolding:
            c = unfold_temporally(cs, t)
            if temporal_interpolation and t < cs.shape[0] - 1:
                c_next = unfold_temporally(cs, t + 1)
        else:
            c = cs[t]  # B X Y C
            if temporal_interpolation and t < cs.shape[0] - 1:
                c_next = cs[t + 1]

        yield t, get_spatial_upscaling_output(decoder, c, taus[t], c_next, corner_pixels, corner_to_point_vectors)

    return


def unfold_temporally(cs: Union[Float[torch.Tensor, "T B X Y C"], List[Float[torch.Tensor, "B X Y C"]]], t: int) -> Float[torch.Tensor, "T B X Y ThreeC"]:
    def get_value_or_zeros(t_i: int) -> Float[torch.Tensor, "T B X Y C"]:
        # Get the correct c if exists, otherwise zeros
        if t_i in range(len(cs)):
            return cs[t_i]
        return torch.zeros_like(cs[0])

    # Return previous, current, and next
    return torch.concat([get_value_or_zeros(t_i) for t_i in [t - 1, t, t + 1]], dim=-1)


def stack_and_maybe_unfold_c_list(c_list: List[Float[torch.Tensor, "B C X Y"]], spatial_unfolding: bool) -> Float[torch.Tensor, "T B X Y C"]:
    cs = torch.stack(c_list, dim=0)  # T B C X Y
    T, _, _, X, _ = cs.shape
    if spatial_unfolding:
        cs = einops.rearrange(cs, "T B C X Y -> (T B) C X Y")
        cs = torch.nn.functional.unfold(cs, kernel_size=(3, 3), padding=(1, 1), stride=1)
        cs = einops.rearrange(cs, "(T B) C (X Y) -> T B C X Y", T=T, X=X)
    # Prepare for linear layer
    return einops.rearrange(cs, "T B C X Y -> T B X Y C")


def accumulate_gradients(model: nn.Module, gradients: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    if gradients is None:
        gradients = [p.grad for p in model.parameters()]
    else:
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                gradients[i] += p.grad
    return gradients


def apply_gradients(model: nn.Module, gradients: List[torch.Tensor]) -> nn.Module:
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            p.grad = gradients[i]
    return model
