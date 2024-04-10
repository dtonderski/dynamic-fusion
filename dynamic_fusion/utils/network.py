from typing import Generator, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import nn

from dynamic_fusion.network_trainer.configuration import SharedConfiguration
from dynamic_fusion.utils.dataset import discretized_events_to_tensors
from dynamic_fusion.utils.datatypes import Batch, TestBatch
from dynamic_fusion.utils.discretized_events import DiscretizedEvents
from dynamic_fusion.utils.superresolution import get_spatial_upscaling_output, get_upscaling_pixel_indices_and_distances


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


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:  # type: ignore[type-arg]
    if isinstance(tensor, np.ndarray):
        return tensor
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


def run_reconstruction(
    encoder: nn.Module,
    decoder: nn.Module,
    discretized_events: DiscretizedEvents,
    device: torch.device,
    config: SharedConfiguration,
    output_shape: Optional[Tuple[int, int]] = None,
    taus_to_evaluate: int = 1,
) -> Float[torch.Tensor, "T C X Y"]:
    with torch.no_grad():
        event_polarity_sum, timestamp_mean, timestamp_std, event_count = discretized_events_to_tensors(discretized_events)
        number_of_temporal_bins = event_polarity_sum.shape[0]
        eps, means, stds, counts = (
            event_polarity_sum.to(device)[None],
            timestamp_mean.to(device)[None],
            timestamp_std.to(device)[None],
            event_count.to(device)[None],
        )

        encoder.reset_states()

        if output_shape is None:
            output_shape = event_polarity_sum.shape[-2:]

        corner_pixels, corner_to_point_vectors = get_upscaling_pixel_indices_and_distances(tuple(eps.shape[-2:]), output_shape)

        taus = np.arange(0, 1, 1 / taus_to_evaluate)
        taus = torch.tensor(taus).to(device)

        reconstructions = []

        # For unfolding, we need 3, for interpolation also, we need 4
        cs_queue = []

        reconstructions = []
        for t in range(number_of_temporal_bins + 2):
            print(f"{t} / {number_of_temporal_bins + 2}", end="\r")
            reconstructions_t = []
            if t < int(number_of_temporal_bins):
                c_t = encoder(
                    eps[:, t], means[:, t] if config.use_mean else None, stds[:, t] if config.use_std else None, counts[:, t] if config.use_count else None
                )
                cs_queue.append(stack_and_maybe_unfold_c_list([c_t], config.spatial_unfolding)[0])

                if t == 0:
                    # Makes code easier if we do it this way
                    cs_queue.insert(0, torch.zeros_like(cs_queue[0]))

            if t > 1:
                # We usually have 4 items in the queue, near the end we have 3 or 2
                c_next = None
                if config.temporal_unfolding:
                    c = unfold_temporally(cs_queue, 1)
                    if config.temporal_interpolation and len(cs_queue) > 2:
                        c_next = unfold_temporally(cs_queue, 2)
                else:
                    c = cs_queue[1]  # B X Y C
                    if config.temporal_interpolation and len(cs_queue) > 2:
                        c_next = cs_queue[2]

                for i_tau in range(len(taus)):
                    if config.spatial_upscaling:
                        r_t = get_spatial_upscaling_output(decoder, c, taus[i_tau : i_tau + 1].to(c), c_next, corner_pixels, corner_to_point_vectors)
                    else:
                        tau = einops.repeat(taus[i_tau : i_tau + 1].to(c), "1 -> T X Y 1", X=c.shape[-3], Y=c.shape[-2])
                        r_t = decoder(torch.concat([c, tau], dim=-1))
                        if c_next is not None:
                            r_tnext = decoder(torch.concat([c_next, tau - 1], dim=-1))
                            r_t = r_t * (1 - tau) + r_tnext * (tau)

                    reconstructions_t.append(to_numpy(r_t))

                reconstructions.append(np.stack(reconstructions_t, axis=0))

            if len(cs_queue) > 3:
                del cs_queue[0]

        reconstruction_stacked = np.stack(reconstructions, axis=0)  # T tau C X Y
        reconstruction_flat = einops.rearrange(reconstruction_stacked, "tau T C D X Y -> (tau T) (C D) X Y")  # D=1
        return reconstruction_flat
