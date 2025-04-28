from typing import (
    List,
    Literal,
    Tuple,
    Union,
)

import torch
from torch import nn

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)


class RandomizedOscillatorsNetwork(nn.Module):
    """
    Randomized Oscillators Network. A recurrent neural network model with
    oscillatory dynamics. The model is defined by the following ordinary
    differential equation:

    .. math::
        \\dot{h} = -\\gamma h - \\epsilon \\dot{h} + \\tanh(W_{in} x + W_{rec} h + b)

    where:
    - :math:`h` is the hidden state,
    - :math:`\\dot{h}` is the derivative of the hidden state,
    - :math:`\\gamma` is the damping factor,
    - :math:`\\epsilon` is the stiffness factor,
    - :math:`W_{in}` is the input-to-hidden weight matrix,
    - :math:`W_{rec}` is the hidden-to-hidden weight matrix,
    - :math:`b` is the bias vector.

    The model is trained by minimizing the mean squared error between the output of the
    model and the target time-series.
    """

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        diffusive_gamma=0.0,
        rho: float = 0.99,
        input_scaling: float = 1.0,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
    ):
        """Initialize the RON model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            dt (float): Time step.
            gamma (float or tuple): Damping factor. If tuple, the damping factor is
                randomly sampled from a uniform distribution between the two values.
            epsilon (float or tuple): Stiffness factor. If tuple, the stiffness factor
                is randomly sampled from a uniform distribution between the two values.
            diffusive_gamma (float): Diffusive term to ensure stability of the forward Euler method.
            rho (float): Spectral radius of the hidden-to-hidden weight matrix.
            input_scaling (float): Scaling factor for the input-to-hidden weight matrix.
                Wrt original paper here we initialize input-hidden in (0, 1) instead of (-2, 2).
                Therefore, when taking input_scaling from original paper, we recommend to multiply it by 2.
            topology (str): Topology of the hidden-to-hidden weight matrix. Options are
                'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz', 'antisymmetric'. Default is
                'full'.
            reservoir_scaler (float): Scaling factor for the hidden-to-hidden weight
                matrix.
            sparsity (float): Sparsity of the hidden-to-hidden weight matrix.
            device (str): Device to run the model on. Options are 'cpu' and 'cuda'.
        """
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        self.diffusive_matrix = diffusive_gamma * torch.eye(n_hid).to(device)
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (gamma_max - gamma_min)
                + gamma_min
            )
        else:
            self.gamma = gamma
        self.gamma = torch.nn.Parameter(self.gamma, requires_grad=False)

        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (eps_max - eps_min)
                + eps_min
            )
        else:
            self.epsilon = epsilon
        self.epsilon = torch.nn.Parameter(self.epsilon, requires_grad=False)

        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        if topology != 'antisymmetric':
            h2h = spectral_norm_scaling(h2h, rho)              
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)

    def cell(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor.
            hy (torch.Tensor): Current hidden state.
            hz (torch.Tensor): Current hidden state derivative.
        """
        hz = hz + self.dt * (
            torch.tanh(
                torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h - self.diffusive_matrix) + self.bias
            )
            - self.gamma * hy
            - self.epsilon * hz
        )

        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass on a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t], hy, hz)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [
            hy
        ]  # list to be compatible with ESN implementation
