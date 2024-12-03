from torch import nn
import torch
import warnings
from numpy import sqrt

from typing import (
    Literal,
)
from acds.archetypes.utils import (
    get_hidden_topology,
)


class TrainedPhysicallyImplementableRandomizedOscillatorsNetwork(nn.Module):
    def __init__(self, n_inp, n_hid, dt, diffusive_gamma=0.0, gamma=1.0, epsilon=1.0, device='cpu',
                 matrix_friction=False, train_oscillators=False,
                 train_recurrent=True, 
                 topology: Literal["orthogonal", "antisymmetric"] = "orthogonal",
                ):
        super().__init__()

        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        self.diffusive_matrix = diffusive_gamma * torch.eye(n_hid).to(device)
        self.matrix_friction = matrix_friction
        self.train_oscillators = train_oscillators

        assert not self.train_oscillators or isinstance(gamma, tuple) and isinstance(epsilon, tuple), \
            "If train_oscillators is True, gamma and epsilon must be tuples."

        if self.matrix_friction and (gamma is not None or epsilon is not None):
            warnings.warn(
                "With epsilon and gamma matrices, the initialization does not follow gamma/epsilon min/max. "
                "It is instead a normal distribution with mean 0 and standard deviation 0.9/sqrt(hidden size)."
            )

        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            if matrix_friction:
                self.gamma = torch.empty(n_hid, n_hid, device=device).normal_(
                    0, 0.9 / sqrt(n_hid))
                self.gamma = torch.abs(self.gamma)
            else:
                self.gamma = torch.rand(n_hid, device=device)
                self.gamma = self.gamma * (gamma_max - gamma_min) + gamma_min

            if matrix_friction:
                self.gamma = torch.matmul(self.gamma.T, self.gamma)
        else:
            self.gamma = gamma

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            if matrix_friction:
                self.epsilon = torch.empty(n_hid, n_hid, device=device).normal_(0, 0.9 / sqrt(n_hid))
                self.epsilon = torch.abs(self.epsilon)
            else:
                self.epsilon = torch.rand(n_hid, device=device)
                self.epsilon = self.epsilon * (eps_max - eps_min) + eps_min

            if matrix_friction:
                self.epsilon = torch.matmul(self.epsilon.T, self.epsilon)
        else:
            self.epsilon = epsilon

        if self.train_oscillators:
            self.gamma = nn.Parameter(self.gamma, requires_grad=True)
            self.epsilon = nn.Parameter(self.epsilon, requires_grad=True)

        if topology == "antisymmetric":
            h2h = get_hidden_topology(n_hid, topology, 0, 0)
        else:
            h2h = torch.empty(n_hid, n_hid, device=device)
            nn.init.orthogonal_(h2h)
        self.h2h = nn.Parameter(h2h, requires_grad=train_recurrent)

        bias = (torch.rand(n_hid) * 2 - 1)
        self.bias = nn.Parameter(bias, requires_grad=True)

        x2h = torch.rand(n_inp, n_hid, device=device)
        self.x2h = nn.Parameter(x2h, requires_grad=True)

    def cell(self, x, hy, hz):
        i2h = torch.matmul(x, self.x2h)
        w = self.h2h - self.diffusive_matrix
        h2h = torch.matmul(hy, w) + self.bias
        h2h_T = torch.transpose(w,0,1)
        
        if self.matrix_friction:
            hz = hz + self.dt * (torch.tanh(i2h) -
                                 torch.matmul(torch.tanh(h2h), h2h_T) -
                                 torch.matmul(hy, self.gamma) - torch.matmul(hz, self.epsilon))
        else:
            hz = hz + self.dt * (torch.tanh(i2h) -
                                 torch.matmul(torch.tanh(h2h), h2h_T) -
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
