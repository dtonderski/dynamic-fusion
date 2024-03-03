from typing import Generator, List, Optional, Tuple
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
    cs: List[Float[torch.Tensor, "T B X Y C"]],
    taus: Float[torch.Tensor, "T B X Y 1"],
    temporal_interpolation: bool,
    temporal_unfolding: bool,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> Generator[Tuple[int, Float[torch.Tensor, "B X Y Cout"]], None, None]:
    t_start = t_start if t_start is not None else 0
    t_end = t_end if t_end is not None else cs[0].shape[0]

    for t in range(t_start, t_end):
        c = cs[t]  # type: ignore  # B X Y C
        c_next = None
        if temporal_interpolation:
            c_next = cs[t + 1]  # type: ignore

        if temporal_unfolding:
            c = torch.concat([cs[t - 1], cs[t], cs[t + 1]], dim=-1)  # type: ignore
            if temporal_interpolation:
                c_next = torch.concat([cs[t], cs[t + 1], cs[t + 2]], dim=-1)  # type: ignore

        r_t = decoder(torch.concat([c, taus[t]], dim=-1))
        if temporal_interpolation:
            r_tnext = decoder(torch.concat([c_next, taus[t] - 1], dim=-1))  # type: ignore
            r_t = r_t * (1 - taus[t]) + r_tnext * (taus[t])
        yield t, r_t
    return


def run_decoder_with_spatial_upscaling(
    decoder: nn.Module,
    cs: Float[torch.Tensor, "T B X Y C"],
    taus: Float[torch.Tensor, "T B"],
    temporal_interpolation: bool,
    temporal_unfolding: bool,
    nearest_pixels: Int[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    start_to_end_vectors: Float[torch.Tensor, "4 XUpscaled YUpscaled 2"],
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> Generator[Tuple[int, Float[torch.Tensor, "B X Y Cout"]], None, None]:
    t_start = t_start if t_start is not None else 0
    t_end = t_end if t_end is not None else cs[0].shape[0]

    for t in range(t_start, t_end):
        c = cs[t]  # B X Y C
        c_next = None
        if temporal_interpolation:
            c_next = cs[t + 1]

        if temporal_unfolding:
            c = torch.concat([cs[t - 1], cs[t], cs[t + 1]], dim=-1)
            if temporal_interpolation:
                c_next = torch.concat([cs[t], cs[t + 1], cs[t + 2]], dim=-1)

        yield t, get_spatial_upscaling_output(decoder, c, taus[t], c_next, nearest_pixels, start_to_end_vectors)

    return


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
