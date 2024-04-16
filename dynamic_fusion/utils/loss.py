from typing import List

import einops
import lpips
import torch
from jaxtyping import Float32
from torch import nn


class UncertaintyLossWithLPIPS(nn.Module):
    loss_function_vgg: lpips.LPIPS
    uncertainty_weight: float
    values: List[float] = []
    weights: List[int] = []

    def __init__(self, uncertainty_weight: float = 1) -> None:
        super().__init__()
        self.loss_function_vgg = lpips.LPIPS(net="vgg")
        self.uncertainty_weight = uncertainty_weight

    def forward(self, x: Float32[torch.Tensor, "B 2 X Y"], y: Float32[torch.Tensor, "B 1 X Y"], epsilon: float = 1e-6) -> Float32[torch.Tensor, " 1"]:
        """x[:,0] is mean, x[:, 1] is log std"""
        means_detached, log_stds = x[:, 0:1].clone().detach(), x[:, 1:2]
        per_pixel_loss = ((means_detached - y) ** 2) / (2 * torch.exp(2 * log_stds) + epsilon) + log_stds

        x = einops.repeat(x[:, 0:1], "B C X Y -> B (C three) X Y", three=3)
        y = einops.repeat(y, "B C X Y -> B (C three) X Y", three=3)
        lpips_loss = self.loss_function_vgg(2 * (x - 0.5), 2 * (y - 0.5))

        return per_pixel_loss.mean() * self.uncertainty_weight + lpips_loss.mean()

    @torch.no_grad()
    def update(self, x: Float32[torch.Tensor, "B C X Y"], y: Float32[torch.Tensor, "B C X Y"]) -> None:
        self.values.append(self(x, y).item())
        self.weights.append(x.numel())

    @torch.no_grad()
    def compute(self) -> float:
        return sum(value * weight / sum(self.weights) for value, weight in zip(self.values, self.weights))

    @torch.no_grad()
    def reset(self) -> None:
        self.values = []
        self.weights = []


class UncertaintyLoss(nn.Module):
    values: List[float] = []
    weights: List[int] = []

    def forward(self, x: Float32[torch.Tensor, "B 2 X Y"], y: Float32[torch.Tensor, "B 1 X Y"], epsilon: float = 1e-6) -> Float32[torch.Tensor, " 1"]:
        """x[:,0] is mean, x[:, 1] is log std"""
        means, log_stds = x[:, 0:1], x[:, 1:2]

        per_pixel_loss = ((means - y) ** 2) / (2 * torch.exp(2 * log_stds) + epsilon) + log_stds

        return per_pixel_loss.mean()

    @torch.no_grad()
    def update(self, x: Float32[torch.Tensor, "B C X Y"], y: Float32[torch.Tensor, "B C X Y"]) -> None:
        self.values.append(self(x, y).item())
        self.weights.append(x.numel())

    @torch.no_grad()
    def compute(self) -> float:
        return sum(value * weight / sum(self.weights) for value, weight in zip(self.values, self.weights))

    @torch.no_grad()
    def reset(self) -> None:
        self.values = []
        self.weights = []


class LPIPS(nn.Module):
    loss_function_vgg: lpips.LPIPS
    values: List[float] = []
    weights: List[int] = []

    def __init__(self, **kwargs: bool) -> None:
        super(LPIPS, self).__init__()
        self.loss_function_vgg = lpips.LPIPS(net="vgg", **kwargs)

    def forward(self, x: Float32[torch.Tensor, "B C X Y"], y: Float32[torch.Tensor, "B C X Y"]) -> Float32[torch.Tensor, " B"]:
        x = einops.repeat(x, "B C X Y -> B (C three) X Y", three=3)
        y = einops.repeat(y, "B C X Y -> B (C three) X Y", three=3)
        return self.loss_function_vgg(2 * (x - 0.5), 2 * (y - 0.5))

    @torch.no_grad()
    def update(self, x: Float32[torch.Tensor, "B C X Y"], y: Float32[torch.Tensor, "B C X Y"]) -> None:
        self.values.append(self(x, y).mean().item())
        self.weights.append(x.numel())

    @torch.no_grad()
    def compute(self) -> float:
        return sum(value * weight / sum(self.weights) for value, weight in zip(self.values, self.weights))

    @torch.no_grad()
    def reset(self) -> None:
        self.values = []
        self.weights = []


def get_reconstruction_loss(loss_name: str, device: torch.device) -> nn.Module:
    if loss_name.upper() == "L1":
        return nn.L1Loss().to(device)
    elif loss_name.upper() == "L2":
        return nn.MSELoss().to(device)
    elif loss_name.upper() == "LPIPS":
        return LPIPS().to(device)
    else:
        raise ValueError(f"Unknown image loss name: {loss_name}")


def get_uncertainty_loss(loss_name: str, device: torch.device, uncertainty_weight: float = 1) -> nn.Module:
    if loss_name.upper() == "UNCERTAINTY":
        return UncertaintyLoss()
    if loss_name.upper() == "UNCERTAINTY_WITH_LPIPS":
        return UncertaintyLossWithLPIPS(uncertainty_weight).to(device)
    raise ValueError(f"Unknown uncertainty loss name: {loss_name}")
