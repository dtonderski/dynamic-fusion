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
        self,
        input_size: int,
        hidden_size: int,
        out_size: int,
        kernel_size: int,
        ra_k: float = 0.95,
        max_t: int = 8,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.hidden_size = hidden_size

        self.state1: Optional[torch.Tensor] = None
        self.state2: Optional[torch.Tensor] = None

        self.input_normalizer = RunningAverageNNZnormalizer(k=ra_k)
        self.time_to_prev_ev = utils.TimeToPrevCounter(max_t=max_t)

        self.head = nn.Conv2d(input_size + 1, hidden_size, kernel_size, 1, padding)
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

        self.reset_states()

    @torch.jit.ignore  # type: ignore
    def reset_states(self) -> None:
        self.input_normalizer.reset_states()
        self.time_to_prev_ev.reset_states()
        self.state1 = None
        self.state2 = None

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: B C X Y
        """
        batch_size, _, imsz0, imsz1 = d.shape
        if self.state1 is None:
            self.state1 = torch.zeros(
                [batch_size, self.hidden_size, imsz0, imsz1]
            ).to(d)

        d_nrm = self.input_normalizer(d)
        time_to_prev = self.time_to_prev_ev(d)
        x_input = torch.concat([d_nrm, time_to_prev], dim=1)

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
        x3_c = torch.concatenate([x3_up, x2, x0], dim=1)

        # resid:
        x4 = self.conv21(x3_c)
        x4_nrm = self.nrm3(x4)
        x4 = F.elu(x4_nrm)
        x4 = self.conv22(x4)
        x4 = F.elu(self.nrm4(x4) + x4_nrm)

        rec: torch.Tensor = self.out_conv(x4)

        return rec
