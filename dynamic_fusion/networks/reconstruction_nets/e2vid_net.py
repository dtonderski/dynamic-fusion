from typing import Optional, Tuple

import torch
from jaxtyping import Shaped
from torch import nn
from torch.nn import functional as F

from dynamic_fusion.networks.layers.normalizers import RunningAverageNNZnormalizer

from .e2vid.unet import UNetRecurrent


class E2VIDRecurrent(nn.Module):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    prev_states = None
    num_encoders: int

    def __init__(self, input_size: int, output_size: int, num_encoders: int = 4) -> None:
        super().__init__()
        self.input_normalizer = RunningAverageNNZnormalizer(k=0.95)
        self.num_encoders = num_encoders

        self.unetrecurrent = UNetRecurrent(  # type: ignore
            num_input_channels=input_size,
            num_output_channels=output_size,
            recurrent_block_type="convlstm",
            activation=None,
            num_encoders=num_encoders,
            base_num_channels=32,
            num_residual_blocks=2,
            norm=None,
            use_upsample_conv=True,
        )

    def forward(self, d: Optional[torch.Tensor], *args: Optional[torch.Tensor]) -> torch.Tensor:
        """
        d: B C X Y
        args: [B C X Y]
        """
        # Tricks here to make sure that the network is compatible with only events, only aps, and aps+events
        # Just need to make sure input_size is correct and everything else should work.
        inputs = torch.concat([tensor for tensor in [d, *args] if tensor is not None], dim=1)
        x_input = self.input_normalizer(inputs)

        x_input, (left, right, top, bottom) = pad_to_divisibility(x_input, 2**self.num_encoders)

        img_pred, states = self.unetrecurrent.forward(x_input, self.prev_states)  # type: ignore

        self.prev_states = []
        for x in states:
            if isinstance(x, tuple):
                self.prev_states.append(tuple(y.clone() for y in x))
            else:
                self.prev_states.append(x.clone())

        return img_pred[:, :, top : img_pred.shape[2] - bottom, left : img_pred.shape[3] - right]  # type: ignore

    @torch.jit.ignore  # type: ignore
    def reset_states(self) -> None:
        self.input_normalizer.reset_states()
        self.prev_states = None


def pad_to_divisibility(
    tensor: Shaped[torch.Tensor, "B C X Y"], target_divisor: int
) -> Tuple[Shaped[torch.Tensor, "B C XPad YPad"], Tuple[int, int, int, int]]:
    height, width = tensor.shape[2], tensor.shape[3]
    pad_height = (target_divisor - height % target_divisor) % target_divisor
    pad_width = (target_divisor - width % target_divisor) % target_divisor
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
    padded_tensor = F.pad(tensor, padding, mode="reflect")
    return padded_tensor, padding
