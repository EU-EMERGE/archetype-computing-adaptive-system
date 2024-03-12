import torch
from torch import nn


class LSTM(nn.Module):
    """LSTM model with a readout layer."""

    def __init__(self, n_inp: int, n_hid: int, n_out: int):
        """Initialize the model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            n_out (int): Number of output units.
        """
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inp, n_hid, batch_first=True, num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shaped as (batch, seq_len, n_inp).

        Returns:
            torch.Tensor: Output tensor, shaped as (batch, n_out).
        """
        out, h = self.lstm(x)
        out = self.readout(out[:, -1])
        return out
