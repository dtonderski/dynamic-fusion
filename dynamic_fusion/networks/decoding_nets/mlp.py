import torch
from torch import nn

from jaxtyping import Float


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError(
                f"n_hidden_layers must be at least 1, but it is {hidden_layers}"
            )

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers - 1):
            self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: Float[torch.Tensor, "*batch C"]) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)  # type: ignore