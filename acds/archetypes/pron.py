import torch
from torch import nn


class PhysicallyImplementableRandomizedOscillatorsNetwork(nn.Module):
    """
    Batch-first (B, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, input_scaling, device='cpu',
                 fading=False):
        super().__init__()

        self.n_hid = n_hid
        self.device = device
        self.fading = fading
        self.dt = dt

        gamma_min, gamma_max = gamma
        self.gamma = torch.rand(n_hid, requires_grad=False, device=device) * (gamma_max - gamma_min) + gamma_min

        eps_min, eps_max = epsilon
        self.epsilon = torch.rand(n_hid, requires_grad=False, device=device) * (eps_max - eps_min) + eps_min

        h2h = torch.empty(n_hid, n_hid, device=device)
        nn.init.orthogonal_(h2h)
        h2h_T = torch.transpose(h2h,0,1)
        self.h2h = nn.Parameter(h2h, requires_grad=False)
        self.h2h_T = nn.Parameter(h2h_T, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz):
        i2h = torch.matmul(x, self.x2h)
        h2h = torch.matmul(hy, self.h2h) + self.bias

        hz = hz + self.dt * (torch.tanh(i2h) -
                             torch.matmul(torch.tanh(h2h), self.h2h_T) -
                             self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x):
        hy = torch.zeros(x.size(0),self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0),self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t],hy,hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [hy]
