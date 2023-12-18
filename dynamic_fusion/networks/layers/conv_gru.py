import torch
from torch import nn
from typing import Type, Union

class ConvGRU(nn.Module):
    """
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        conv_layer: Union[Type[nn.Conv1d], Type[nn.Conv2d]],
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.hidden_size = hidden_size

        self.reset_gate = conv_layer(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.update_gate = conv_layer(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.out_gate = conv_layer(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )

        nn.init.orthogonal_(self.reset_gate.weight)  # type: ignore
        nn.init.orthogonal_(self.update_gate.weight)  # type: ignore
        nn.init.orthogonal_(self.out_gate.weight)  # type: ignore
        nn.init.constant_(self.reset_gate.bias, 0.0)  # type: ignore
        nn.init.constant_(self.update_gate.bias, 0.0)  # type: ignore
        nn.init.constant_(self.out_gate.bias, 0.0)  # type: ignore

    def forward(self, input_: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(
            self.out_gate(torch.cat([input_, prev_state * reset], dim=1))
        )
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state  # type: ignore
