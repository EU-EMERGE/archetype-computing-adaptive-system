from torch import nn
import torch
import warnings
from numpy import sqrt


class hcoRNN(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, device='cpu',
                 matrix_friction=False, train_oscillators=False):
        super().__init__()

        self.n_hid = n_hid
        self.device = device
        self.dt = dt
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
                # self.gamma = torch.abs(self.gamma)
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
                # self.epsilon = torch.abs(self.epsilon)
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

        self.i2h = nn.Linear(n_inp, n_hid)
        self.h2h = nn.Linear(n_hid, n_hid, bias=False)

    def cell(self, x, hy, hz):
        if self.matrix_friction:
            hz = hz + self.dt * (torch.tanh(self.i2h(x) + self.h2h(hy)) -
                                 torch.matmul(hy, self.gamma) - torch.matmul(hz, self.epsilon))
        else:
            hz = hz + self.dt * (torch.tanh(self.i2h(x) + self.h2h(hy)) -
                                 self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x):
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        all_states = []

        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t],hy,hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [hy]
