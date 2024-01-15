import einops
import lpips
import torch
from jaxtyping import Float32
from torch import nn


class LPIPS(nn.Module):
    loss_function_vgg: lpips.LPIPS

    def __init__(self) -> None:
        super(LPIPS, self).__init__()
        self.loss_function_vgg = lpips.LPIPS(net="vgg")

    def forward(
        self, x: Float32[torch.Tensor, "B C X Y"], y: Float32[torch.Tensor, "B C X Y"]
    ) -> Float32[torch.Tensor, " B"]:
        x = einops.repeat(x, "B C X Y -> B (C three) X Y", three=3)
        y = einops.repeat(y, "B C X Y -> B (C three) X Y", three=3)
        return self.loss_function_vgg(2 * (x - 0.5), 2 * (y - 0.5))