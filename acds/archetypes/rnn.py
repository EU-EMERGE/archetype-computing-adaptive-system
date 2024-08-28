import torch
from torch import nn
import numpy as np


class RNN_DFA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grad_clip=5, device='cpu', truncation=None):
        super().__init__()

        self.hidden_size = hidden_size

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.V1 = nn.Linear(input_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=True)
        self.BW1 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)
        self.BV1 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)
        self.grad_clip = grad_clip
        self.device = device
        self.output_size = output_size
        self.truncation = truncation

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=self.device)

    @torch.no_grad()
    def forward(self, x, y=None):
        hidden = self.initHidden(x.shape[0])

        act_list = [hidden]  # will have len=seq_len+1

        for t in range(x.size(1)):
            preact = self.V1(x[:, t]) + self.W1(hidden)
            hidden = torch.tanh(preact)

            if y is not None:
                act_list.append(hidden)

        output = torch.softmax(self.Wout(hidden), dim=-1)

        if y is not None:
            truncation = x.size(1) if self.truncation is None else self.truncation

            error = output - torch.nn.functional.one_hot(y, num_classes=self.output_size).float()
            Be = torch.matmul(error, self.BW1)
            B2e = torch.matmul(error, self.BV1)

            dW = torch.zeros(x.size(0), self.hidden_size, self.hidden_size, device=self.device)
            dV = torch.zeros(x.size(0), self.hidden_size, x.size(-1), device=self.device)
            dbW = torch.zeros(x.size(0), self.hidden_size, device=self.device)

            for t in list(reversed(range(x.size(1))))[:truncation]:
                derivative = 1 - act_list[t] ** 2
                # batched outer product
                dW += torch.einsum('bh,bH->bhH', Be * derivative, act_list[t-1])
                dV += torch.einsum('bh,bi->bhi', B2e * derivative, x[:, t])
                dbW += Be * derivative

            # normalize by batch size
            dW = torch.mean(dW, dim=0)
            dV = torch.mean(dV, dim=0)
            dbW = torch.mean(dbW, dim=0)

            dW /= torch.norm(dW)
            dV /= torch.norm(dV)
            dbW /= torch.norm(dbW)

            dW = torch.clip(dW, min=-self.grad_clip, max=self.grad_clip)
            dV = torch.clip(dV, min=-self.grad_clip, max=self.grad_clip)
            dbW = torch.clip(dbW, min=-self.grad_clip, max=self.grad_clip)

            return output, hidden, dW, dV, dbW, error
        else:
            return output

    def compute_update(self, last_hidden, dW, dV, db, error):
        for name, p in self.named_parameters():
            if 'Wout.weight' in name:
                new_value = torch.matmul(error.T, last_hidden) / float(last_hidden.size(0))
                p.grad = new_value.clone()
            elif 'Wout.bias' in name:
                new_value = error.mean(dim=0)
                p.grad = new_value.clone()
            elif 'W1.weight' in name:
                p.grad = dW.clone()
            elif 'W1.bias' in name:
                p.grad = db.clone()
            elif 'V1.weight' in name:
                p.grad = dV.clone()


class GRU_DFA(RNN_DFA):
    def __init__(self, input_size, hidden_size, output_size, grad_clip=5, device='cpu', truncation=None):
        super().__init__(input_size, hidden_size, output_size, grad_clip, device, truncation)
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V1 = nn.Linear(input_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.V2 = nn.Linear(input_size, hidden_size, bias=False)
        self.W3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.V3 = nn.Linear(input_size, hidden_size, bias=False)
        self.BW2 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)
        self.BV2 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)
        self.BW3 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)
        self.BV3 = torch.randn(output_size, hidden_size, device=device) / np.sqrt(hidden_size)

    @torch.no_grad()
    def forward(self, x, y=None):
        hidden = self.initHidden(x.shape[0])

        rs_t, zs_t, hs_t, hiddens = [], [], [], [hidden]
        for t in range(x.size(1)):
            z = torch.sigmoid(self.W2(hidden) + self.V2(x[:, t]))
            r = torch.sigmoid(self.W3(hidden) + self.V3(x[:, t]))
            h = torch.tanh(self.V1(x[:, t]) + self.W1(hidden * r))
            hidden = ((1 - z) * h) + (z * hidden)
            rs_t.append(r)
            zs_t.append(z)
            hs_t.append(h)
            hiddens.append(hidden)

        output = torch.softmax(self.Wout(hidden), dim=-1)

        if y is not None:
            truncation = x.size(1) if self.truncation is None else self.truncation

            error = output - torch.nn.functional.one_hot(y, num_classes=self.output_size).float()
            BW1e = torch.matmul(error, self.BW1)
            BV1e = torch.matmul(error, self.BV1)
            BW2e = torch.matmul(error, self.BW2)
            BV2e = torch.matmul(error, self.BV2)
            BW3e = torch.matmul(error, self.BW3)
            BV3e = torch.matmul(error, self.BV3)

            dW = [torch.zeros(x.size(0), self.hidden_size, self.hidden_size, device=self.device) for _ in range(3)]
            dV = [torch.zeros(x.size(0), self.hidden_size, x.size(-1), device=self.device) for _ in range(3)]
            db = [torch.zeros(x.size(0), self.hidden_size, device=self.device) for _ in range(3)]

            for t in list(reversed(range(x.size(1))))[:truncation]:
                zgate = ((BV2e * hiddens[t-1]) + (-BV2e * hs_t[t])) * (rs_t[t] * (1 - rs_t[t]))
                dV[1] += torch.einsum('bh,bi->bhi', zgate, x[:, t])
                dW[1] += torch.einsum('bh,bH->bhH', ((BW2e * hiddens[t-1]) + (-BW2e * hs_t[t])) * (rs_t[t] * (1 - rs_t[t])), hiddens[t-1])
                db[1] += zgate
                rgate = (self.W1((BV3e * (1-zs_t[t])) * (1-hs_t[t]**2)) * hiddens[t-1]) * (rs_t[t] * (1 - rs_t[t]))
                dV[2] += torch.einsum('bh,bi->bhi', rgate, x[:, t])
                dW[2] += torch.einsum('bh,bH->bhH', (self.W1((BW3e * (1-zs_t[t])) * (1-hs_t[t]**2)) * hiddens[t-1]) * (rs_t[t] * (1 - rs_t[t])), hiddens[t-1])
                db[2] += rgate
                hgate = self.W1((BV1e * (1-zs_t[t]))) * (1-hs_t[t]**2)
                dV[0] += torch.einsum('bh,bi->bhi', hgate, x[:, t])
                dW[0] += torch.einsum('bh,bH->bhH', self.W1((BW1e * (1-zs_t[t]))) * (1-hs_t[t]**2), rs_t[t] * hiddens[t-1])
                db[0] += hgate

            # normalize by batch size
            dV = [el.mean(dim=0) for el in dV]
            dW = [el.mean(dim=0) for el in dW]
            db = [el.mean(dim=0) for el in db]

            dV = [el / torch.norm(el) for el in dV]
            dW = [el / torch.norm(el) for el in dW]
            db = [el / torch.norm(el) for el in db]

            dV = [torch.clip(el, min=-self.grad_clip, max=self.grad_clip) for el in dV]
            dW = [torch.clip(el, min=-self.grad_clip, max=self.grad_clip) for el in dW]
            db = [torch.clip(el, min=-self.grad_clip, max=self.grad_clip) for el in db]

            return output, hidden, dW, dV, db, error
        else:
            return output

    def compute_update(self, last_hidden, dW, dV, db, error):
        for name, p in self.named_parameters():
            if 'Wout.weight' in name:
                new_value = torch.matmul(error.T, last_hidden) / float(last_hidden.size(0))
                p.grad = new_value.clone()
            elif 'Wout.bias' in name:
                new_value = error.mean(dim=0)
                p.grad = new_value.clone()
            elif 'W1.weight' in name:
                p.grad = dW[0].clone()
            elif 'V1.bias' in name:
                p.grad = db[0].clone()
            elif 'V1.weight' in name:
                p.grad = dV[0].clone()
            elif 'V2.weight' in name:
                p.grad = dV[1].clone()
            elif 'V3.weight' in name:
                p.grad = dV[2].clone()
            elif 'W2.weight' in name:
                p.grad = dW[1].clone()
            elif 'W3.weight' in name:
                p.grad = dW[2].clone()


class LSTM(nn.Module):
    """LSTM model with a readout layer."""

    def __init__(self, n_inp: int, n_hid: int, n_out: int, gru: bool = False, rnn: bool = False):
        """Initialize the model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            n_out (int): Number of output units.
            gru (bool, optional): Use GRU instead of LSTM. Defaults to False.
            rnn (bool, optional): Use RNN instead of LSTM. Defaults to False.
        """
        super().__init__()
        if gru:
            self.rnn = torch.nn.GRU(n_inp, n_hid, batch_first=True, num_layers=1)
        elif rnn:
            self.rnn = torch.nn.RNN(n_inp, n_hid, batch_first=True, num_layers=1)
        else:
            self.rnn = torch.nn.LSTM(n_inp, n_hid, batch_first=True, num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shaped as (batch, seq_len, n_inp).

        Returns:
            torch.Tensor: Output tensor, shaped as (batch, n_out).
        """
        out, h = self.rnn(x)
        out = self.readout(out[:, -1])
        return out
