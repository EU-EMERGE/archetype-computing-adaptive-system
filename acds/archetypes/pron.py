import torch
from torch import nn


def get_input_fn(fn_type, n_inp, n_hid, input_scaling, device):
    if fn_type == 'linear':
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        x2h = nn.Parameter(x2h, requires_grad=False).to(device)
        return x2h
    elif fn_type == 'mlp':
        h = 256
        mlp = nn.Sequential(
            nn.Linear(n_inp, h),
            nn.Tanh(),
            nn.Linear(h, n_hid)
        )
        return mlp.to(device)
    else:
        raise ValueError(f"Unknown input function type: {fn_type}")


class PhysicallyImplementableRandomizedOscillatorsNetwork(nn.Module):
    """
    Batch-first (BW1, L, I)
    """
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, input_scaling, device='cpu',
                 input_function='linear', matrix_friction=False):
        """

        Args:
            n_inp:
            n_hid:
            dt:
            gamma:
            epsilon:
            input_scaling:
            device:
        """
        super().__init__()

        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        self.input_function = input_function
        self.matrix_friction = matrix_friction

        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            if matrix_friction:
                self.gamma = torch.empty(n_hid, n_hid, requires_grad=False, device=device).normal_(0, 0.9 / torch.sqrt(torch.tensor(n_hid)))
                self.gamma = torch.abs(self.gamma)
            else:
                self.gamma = torch.rand(n_hid, requires_grad=False, device=device)
                self.gamma = self.gamma * (gamma_max - gamma_min) + gamma_min

            if matrix_friction:
                self.gamma = torch.matmul(self.gamma.T, self.gamma)
        else:
            self.gamma = gamma

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            if matrix_friction:
                self.epsilon = torch.empty(n_hid, n_hid, requires_grad=False, device=device).normal_(0, 0.9 / torch.sqrt(torch.tensor(n_hid)))
                self.epsilon = torch.abs(self.epsilon)
            else:
                self.epsilon = torch.rand(n_hid, requires_grad=False, device=device)
                self.epsilon = self.epsilon * (eps_max - eps_min) + eps_min

            if matrix_friction:
                self.epsilon = torch.matmul(self.epsilon.T, self.epsilon)
        else:
            self.epsilon = epsilon

        h2h = torch.empty(n_hid, n_hid, device=device)
        nn.init.orthogonal_(h2h)
        h2h_T = torch.transpose(h2h,0,1)
        self.h2h = nn.Parameter(h2h, requires_grad=False)
        self.h2h_T = nn.Parameter(h2h_T, requires_grad=False)

        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

        self.x2h = get_input_fn(input_function, n_inp, n_hid, input_scaling, device)

    def cell(self, x, hy, hz):
        if self.input_function == 'linear':
            i2h = torch.matmul(x, self.x2h)
        else:
            i2h = self.x2h(x)

        h2h = torch.matmul(hy, self.h2h) + self.bias

        if self.matrix_friction:
            hz = hz + self.dt * (torch.tanh(i2h) -
                                 torch.matmul(torch.tanh(h2h), self.h2h_T) -
                                 torch.matmul(hy, self.gamma) - torch.matmul(hz, self.epsilon))
        else:
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


class MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, input_scaling, device='cpu'):
        super().__init__()

        assert n_hid % 2 == 0, "n_hid must be even"

        self.n_hid = n_hid
        self.device = device
        self.dt = dt

        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = (
                    torch.rand(n_hid, requires_grad=False, device=device)
                    * (gamma_max - gamma_min)
                    + gamma_min
            )
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = (
                    torch.rand(n_hid, requires_grad=False, device=device)
                    * (eps_max - eps_min)
                    + eps_min
            )
        else:
            self.epsilon = epsilon

        h2ht = torch.empty(int(n_hid/2), int(n_hid/2), device=device)
        h2hl = torch.empty(int(n_hid/2), int(n_hid/2), device=device)
        nn.init.orthogonal_(h2ht)
        nn.init.orthogonal_(h2hl)
        h2htinv = h2ht.T
        h2hlinv = h2hl.T

        h2h = torch.zeros(n_hid, n_hid, device=device)
        h2h[:int(n_hid/2), :int(n_hid/2)] = h2ht
        h2h[int(n_hid/2):, int(n_hid/2):] = h2hl

        h2hinv = torch.zeros(n_hid, n_hid, device=device)
        h2hinv[:int(n_hid/2), :int(n_hid/2)] = h2htinv
        h2hinv[int(n_hid/2):, int(n_hid/2):] = h2hlinv

        self.h2h = nn.Parameter(h2h, requires_grad=False)
        self.h2hinv = nn.Parameter(h2hinv, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(self, x, hy, hz):
        i2h = torch.matmul(x, self.x2h)
        h2h = torch.matmul(hy, self.h2h) + self.bias

        hz = hz + self.dt * (torch.tanh(i2h) -
                             torch.matmul(torch.tanh(h2h), -self.h2hinv) -
                             self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x):
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t], hy, hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [hy]

