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
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (eps_max - eps_min)
                + eps_min
            )
        else:
            self.epsilon = epsilon

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
        # convert to same type of x
        hz = hz.to(x.dtype)
        hy = hy.to(x.dtype)
        
        hz = hz + self.dt * (
            torch.tanh(
                torch.matmul(x, self.x2h.to(dtype=x.dtype)) + torch.matmul(hy, self.h2h.to(dtype=x.dtype) - self.diffusive_matrix.to(dtype=x.dtype)) + self.bias.to(dtype=x.dtype)
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


class DeepRandomizedOscillatorsNetwork(nn.Module):
    """
    Deep Randomized Oscillators Network, using two stack of RON model one over another developing depth
    over.
    
    A recurrent deep neural network model with
    oscillatory dynamics stacked in layers. The model is defined by the following ordinary
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
        total_units: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        n_layers: int = 1,
        diffusive_gamma=0.0,
        rho: float = 0.99,
        input_scaling: float = 1.0,
        inter_scaling: float = 1.0,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cuda",
        concat: bool = True,
        # TODO implement sparse connectivity later...
        connectivity_input: int = 10,
        connectivity_inter: int = 10,
    ):
        """Initialize the DeepRON model.

        Args:
            n_inp (int): number of input units. Default to 1
            total_units (int): Total number of neurons in RON.
            dt (float): Time step.
            n_layers (int): Number of layers in the network.
            concat: (bool): If True, the output of each layer is concatenated. If False, only the output of the last layer is returned.
        """
        super().__init__()
        self.inter_scaling = inter_scaling
        self.n_layers = n_layers
        self.total_units = total_units
        self.reservoir_scaler = reservoir_scaler
        # if True, then the input and output tensors are provided as (batch, seq, feature)
        #self.batch_first = True
        
        self.layers = nn.ModuleList()   
        
        self.concat = concat

        if concat:
            self.layer_units = int(total_units / n_layers) 
        else:
            self.layer_units = total_units
            
        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter
        
        deepron_layers = [
            RandomizedOscillatorsNetwork(
                n_inp=n_inp, n_hid=self.layer_units + total_units % n_layers,
                                    input_scaling=input_scaling_others,
                                    dt=dt,
                                    gamma=gamma,
                                    epsilon=epsilon,
                                    reservoir_scaler=self.reservoir_scaler
                                    #TODO still sparse connectivity to implement
                                    #connectivity_input=connectivity_input_1,
                                    #connectivity_recurrent=connectivity_input_others,
            )
        ]
            
        last_h_size = self.layer_units + total_units % n_layers
        
        for _ in range(n_layers - 1):
            deepron_layers.append(
                RandomizedOscillatorsNetwork(
                    n_inp=last_h_size, n_hid=self.layer_units,
                    input_scaling=input_scaling_others,
                    dt=dt,
                    gamma=gamma,
                    epsilon=epsilon,
                    #connectivity_input=connectivity_input_others,
                    #connectivity_recurrent=connectivity_recurrent,
                )
            )
            last_h_size = self.layer_units
        self.ron_reservoir = nn.ModuleList(deepron_layers)
        
    
    def forward(self, hy: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass on the layers of the DeepRON a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        # list to store the last state of each layer
        layer_states = []
        # list to store the hidden states of each layer
        states = []
        
        for _, ron_layer in enumerate(self.ron_reservoir):
            [hy, last_state] = ron_layer(hy)
            states.append(hy)
            layer_states.append(last_state[0])
        
        states_uncat = states
        
        if self.concat:
            # check what dim we need to concat
            hy = torch.cat(states, dim=2)
        else:
            # if not concat, return only the last layer
            hy = states[-1]
            
       # Choose if return all_states from all layers for the  
       
        return hy, layer_states, states_uncat
        