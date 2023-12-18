from typing import Tuple
import torch
import einops


def nnz_mean_std_2d(d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    d: B C H W
    """
    mask = (d != 0.0).float()  # type: ignore
    # count = mask.sum(dim=(2, 3), keepdim=True)
    count = einops.reduce(mask, "b c h w -> b c 1 1", "sum")
    count = count.clamp(min=1.0)

    # mean = d.sum(dim=(2, 3), keepdim=True) / count
    mean = einops.reduce(d, "b c h w -> b c 1 1", "sum") / count

    d_c = (d - mean) * mask
    # std = (d_c ** 2).sum(dim=(2, 3), keepdim=True) / count
    std = einops.reduce(d_c**2, "b c h w -> b c 1 1", "sum") / count
    std = torch.sqrt(std.clamp(min=1e-6))

    return mean, std


def nnz_mean_std_1d(d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    d: B C H
    """
    mask = (d != 0.0).float()  # type: ignore
    count = einops.reduce(mask, "b c h -> b c 1", "sum")
    count = count.clamp(min=1.0)
    mean = einops.reduce(d, "b c h -> b c 1", "sum") / count
    d_c = (d - mean) * mask
    std = einops.reduce(d_c**2, "b c h -> b c 1", "sum") / count
    std = torch.sqrt(std.clamp(min=1e-6))

    return mean, std


class RunningAverageNNZnormalizer(torch.nn.Module):
    def __init__(self, k: float, k_trainable: bool = False) -> None:
        super().__init__()

        self.first = True

        self.k = torch.nn.Parameter(
            data=torch.tensor(k, dtype=torch.float), requires_grad=k_trainable
        )
        self.ra_mean = torch.zeros(1, dtype=torch.float)
        self.ra_std = torch.ones(1, dtype=torch.float)

    def reset_states(self) -> None:
        self.first = True
        self.ra_mean = torch.zeros(1, dtype=torch.float)
        self.ra_std = torch.ones(1, dtype=torch.float)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: B C H W
        """
        if d.ndim == 4:
            mean, std = nnz_mean_std_2d(d)
        elif d.ndim == 3:
            mean, std = nnz_mean_std_1d(d)
        else:
            raise ValueError("Wrong input size")

        if self.first:
            self.ra_mean = mean
            self.ra_std = std
            self.first = False
        else:
            self.ra_mean = self.k * self.ra_mean + (1 - self.k) * mean
            self.ra_std = self.k * self.ra_std + (1 - self.k) * std

        mask = (d != 0.0).float()  # type: ignore
        d = (d - self.ra_mean) / self.ra_std * mask

        return d


class InstanceNorm2dPlus(torch.nn.Module):
    def __init__(self, num_features: int, bias: bool = True) -> None:
        super().__init__()

        self.num_features = num_features
        self.bias = bias

        self.instance_norm = torch.nn.InstanceNorm2d(
            num_features=num_features, affine=False, track_running_stats=False
        )
        self.alpha = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)

        if bias:
            self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: B C H W
        """

        means = torch.mean(d, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))  # type: ignore

        h = self.instance_norm(d)
        h += means[..., None, None] * self.alpha[..., None, None]

        out = self.gamma.view(-1, self.num_features, 1, 1) * h
        if self.bias:
            out += self.beta.view(-1, self.num_features, 1, 1)

        return out  # type: ignore
