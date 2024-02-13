import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inp, n_hid, batch_first=True,
                                  num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.readout(out[:, -1])
        return out