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
        n_hid: int,
        n_hid_layers: List[int],
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        # TODO if need add 
        # tot_units: int = 500,
        diffusive_gamma=0.0,
        rho: float = 0.99,
        input_scaling: float = 1.0,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cuda",
        concat: bool = True,
        # maybe add and other sparse connectivity later
        # connectivity_inter: int = 10
    ):
        super().__init__()
        
        # check if CUDA is on
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda" 
        
        self.layers = nn.ModuleList()   
        # init first layer density
        input_dim = n_inp
        
        self.concat = concat
        # what means batch_first?
        # if True, then the input and output tensors are provided as (batch, seq, feature)
        # TODO should we add this?
        #self.batch_first = True
        
        
        ## -- Layer units questions --
        # TODO maybe to be in line with DeepReservoir we should do this
        # if concat layers_unit = tot_units//len(n_hid_layers)
        # else layers_unit = tot_units
        # for now we give a list of integer represeting the number of units in each layer
        # Does something changes if different layers have different number of units? 
        # Theorically yes
        ## end
        
        for n_hid in n_hid_layers:
            self.layers.append(RandomizedOscillatorsNetwork(input_dim, n_hid, dt, gamma, epsilon, 
                                                    diffusive_gamma, rho, input_scaling, 
                                                    topology, reservoir_scaler, sparsity, device
                                                    )
            )
            # next size of hidden layer
            input_dim = n_hid 
    
    
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
        
        for layer in self.layers:
            hy, last_state = layer(hy)
            states.append(hy)
            layer_states.append(last_state[0])
        
        if self.concat:
            # check what dim we need to concat
            hy = torch.cat(states, dim=2)
        else:
            # if not concat, return only the last layer
            hy = states[-1]
            
        # TODO: Debug to check if the shapes are correct
        # with a 2 layer model we should get if concat is True
        # torch.Size([1, 100, 200])
        # else torch.Size([1, 100, 100])
        print("Shape of the output", hy.shape)
        print("Shape of layer states", [state.shape for state in states])
        print("Shape of the last state of each layer", [state.shape for state in layer_states])
        print("Shape of the last state of the last layer", layer_states[-1].shape)
       
        # TODO should we return as list 
        return hy, layer_states
        