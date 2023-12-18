from typing import Optional
import torch

class TimeToPrevCounter(torch.nn.Module):

    def __init__(self, max_t: int = 8) -> None:
        super().__init__()

        self.max_t = torch.nn.Parameter(
            data=torch.tensor(max_t, dtype=torch.int),
            requires_grad=False
        )
        self.time_to_prev: Optional[torch.Tensor] = None

    def reset_states(self) -> None:
        self.time_to_prev = None

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: B C H W
        """
        if self.time_to_prev is None:
            self.time_to_prev = torch.zeros_like(d)

        mask = (d == 0).float()  # type: ignore
        count = (self.time_to_prev + 1.) * mask
        count = count.clamp(max=self.max_t)
        self.time_to_prev = count

        return count / self.max_t  # type: ignore
