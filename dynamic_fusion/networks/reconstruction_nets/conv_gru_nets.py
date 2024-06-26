from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import dynamic_fusion.networks.reconstruction_nets.utils as utils
from dynamic_fusion.networks.layers.conv_gru import ConvGRU

# TODO: try using relative imports for these as in the same sub-package? (more portable?)
from dynamic_fusion.networks.layers.normalizers import (
    InstanceNorm2dPlus,
    RunningAverageNNZnormalizer,
)


class ConvGruNetV1(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, out_size: int, kernel_size: int, ra_k: float = 0.95, max_t: int = 8, use_time_to_prev_ev: bool = True, old_norm: bool = True
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.hidden_size = hidden_size

        self.state1: Optional[torch.Tensor] = None
        self.state2: Optional[torch.Tensor] = None

        self.input_normalizer = RunningAverageNNZnormalizer(k=ra_k)
        self.time_to_prev_ev = utils.TimeToPrevCounter(max_t=max_t)

        self.head = nn.Conv2d(input_size + use_time_to_prev_ev, hidden_size, kernel_size, 1, padding)
        self.nrm_head = InstanceNorm2dPlus(hidden_size)

        self.conv11 = nn.Conv2d(hidden_size, hidden_size, kernel_size, 1, padding)
        self.conv12 = nn.Conv2d(hidden_size, hidden_size, kernel_size, 1, padding)
        self.conv12_ds = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2)
        self.conv21 = nn.Conv2d(hidden_size * 3, hidden_size, kernel_size, 1, padding)
        self.conv22 = nn.Conv2d(hidden_size, hidden_size, kernel_size, 1, padding)

        self.nrm1 = InstanceNorm2dPlus(hidden_size)
        self.nrm2 = InstanceNorm2dPlus(hidden_size)
        self.nrm3 = InstanceNorm2dPlus(hidden_size)
        self.nrm4 = InstanceNorm2dPlus(hidden_size)

        self.net1 = ConvGRU(hidden_size, hidden_size, kernel_size, nn.Conv2d)
        self.net2 = ConvGRU(hidden_size, hidden_size, kernel_size, nn.Conv2d)

        self.out_conv = nn.Conv2d(hidden_size, out_size, kernel_size, padding=padding)

        self.old_norm = old_norm

        self.reset_states()

    @torch.jit.ignore  # type: ignore
    def reset_states(self) -> None:
        self.input_normalizer.reset_states()
        self.time_to_prev_ev.reset_states()
        self.state1 = None
        self.state2 = None

    def forward(self, d: Optional[torch.Tensor], *args: Optional[torch.Tensor]) -> torch.Tensor:
        """
        d: B C X Y
        args: [B C X Y]
        """
        # Tricks here to make sure that the network is compatible with only events, only aps, and aps+events
        # Just need to make sure input_size is correct and everything else should work.
        if d is not None:
            batch_size, _, imsz0, imsz1 = d.shape
            time_to_prev = self.time_to_prev_ev(d)
            tensor_like = d
        else:
            tensor_like = [tensor for tensor in args if tensor is not None][0]
            batch_size, _, imsz0, imsz1 = tensor_like.shape
            time_to_prev = None

        if self.old_norm:
            d_nrm = self.input_normalizer(d) if d is not None else None
            x_input = torch.concat(
                [tensor for tensor in [d_nrm, time_to_prev, *args] if tensor is not None],
                dim=1,
            )
        else:
            inputs = torch.concat([tensor for tensor in [d, time_to_prev, *args] if tensor is not None], dim=1)
            x_input = self.input_normalizer(inputs)

        if self.state1 is None:
            self.state1 = torch.zeros([batch_size, self.hidden_size, imsz0, imsz1]).to(tensor_like)

        x0 = self.head(x_input)
        x0 = F.elu(self.nrm_head(x0))

        x1 = self.net1(x0, self.state1)
        self.state1 = x1.clone()

        # resid:
        x2 = self.conv11(x1)
        x2 = F.elu(self.nrm1(x2))
        x2 = self.conv12(x2)
        x2 = F.elu(self.nrm2(x2) + x1)

        x2_ds = F.elu(self.conv12_ds(x2))
        if self.state2 is None:
            self.state2 = torch.zeros_like(x2_ds)
        x3 = self.net2(x2_ds, self.state2)
        self.state2 = x3.clone()

        x3_up = F.interpolate(x3, size=(imsz0, imsz1), mode="bilinear")
        x3_c = torch.concat([x3_up, x2, x0], dim=1)

        # resid:
        x4 = self.conv21(x3_c)
        x4_nrm = self.nrm3(x4)
        x4 = F.elu(x4_nrm)
        x4 = self.conv22(x4)
        x4 = F.elu(self.nrm4(x4) + x4_nrm)

        rec: torch.Tensor = self.out_conv(x4)

        return rec
