from typing import Optional

import numpy as np
import torch
from torch import nn

from acds.archetypes.utils import (
    sparse_eye_init,
    sparse_recurrent_tensor_init,
    sparse_tensor_init,
    spectral_norm_scaling,
)


class ReservoirCell(torch.nn.Module):
    """Shallow reservoir to be used as cell of a Recurrent Neural Network. The
    equation of the reservoir is given by:

    .. math::
        h_t = (1 - \\alpha) h_{t-1} + \\alpha \\tanh(W_{in} x_t + W_{rec} h_{t-1} + b)

    where:
    - :math:`h_t` is the hidden state at time t,
    - :math:`x_t` is the input at time t,
    - :math:`W_{in}` is the input weight matrix,
    - :math:`W_{rec}` is the recurrent weight matrix,
    - :math:`b` is the bias,
    - :math:`\\alpha` is the leaking rate.

    The implementation is derivated from the one in https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py

    If you use this code in your work, please cite the following paper, in which the
    concept of Deep Reservoir Computing has been introduced:

    Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
    A  critical  experimental  analysis.    Neurocomputing268,  87-99  (2017).
    https://doi.org/10.1016/j.neucom.2016.12.08924.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_input: int = 10,
        connectivity_recurrent: int = 10,
    ):
        """Initializes the ReservoirCell.

        Args:
            input_size (int): number of input units.
            units (int): number of recurrent neurons in the reservoir.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
        """
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent

        self.kernel = (
            sparse_tensor_init(input_size, self.units, self.connectivity_input)
            * self.input_scaling
        )
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

        W = sparse_recurrent_tensor_init(self.units, C=self.connectivity_recurrent)
        # re-scale the weight matrix to control the effective spectral radius
        # of the linearized system
        if self.leaky == 1:
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        else:
            I = sparse_eye_init(self.units)
            W = W * self.leaky + (I * (1 - self.leaky))
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = (W + I * (self.leaky - 1)) * (1 / self.leaky)
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)

        # uniform init in [-1, +1] times input_scaling
        self.bias = (torch.rand(self.units) * 2 - 1) * self.input_scaling
        self.bias = nn.Parameter(self.bias, requires_grad=False)

    def forward(self, xt: torch.Tensor, h_prev: torch.Tensor):
        """Computes the output of the cell given the input and previous state.

        Args:
            xt (torch.Tensor): input tensor shaped as (batch, time, input_dim).
            h_prev (torch.Tensor): previous state tensor shaped as
                (batch, time, state_dim).
        Returns:
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
        """
        input_part = torch.mm(xt, self.kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)

        output = torch.tanh(input_part + self.bias + state_part)
        leaky_output = h_prev * (1 - self.leaky) + output * self.leaky
        return leaky_output, leaky_output


class ReservoirLayer(torch.nn.Module):
    """Shallow reservoir to be used as Recurrent Neural Network layer.

    The layer is composed by a number of ReservoirCell, each of which is used to process
    the input and the previous state at each time step.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_input: int = 10,
        connectivity_recurrent: int = 10,
    ):
        """Initializes the ReservoirLayer.

        Args:
            input_size (int): number of input units.
            units (int): number of recurrent neurons in the reservoir.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
        """
        super().__init__()
        self.net = ReservoirCell(
            input_size,
            units,
            input_scaling,
            spectral_radius,
            leaky,
            connectivity_input,
            connectivity_recurrent,
        )

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state to zeros.

        Args:
            batch_size (int): size of the batch.
        Returns:
            torch.Tensor: hidden state tensor shaped as (batch_size, state_dim).
        """
        return torch.zeros(batch_size, self.net.units)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        """Computes the output of the cell given the input and previous state.

        Args:
            x (torch.Tensor): input tensor shaped as (batch, time, input_dim).
            h_prev (torch.Tensor): previous state tensor shaped as
                (batch, time, state_dim). If None, the hidden state is initialized
                to zeros. Defaults to None.
        Returns:
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
        """

        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0]).to(x.device)

        hs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            _, h_prev = self.net(xt, h_prev)
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev


class DeepReservoir(torch.nn.Module):
    """Deep Reservoir to be used as Recurrent Neural Network.

    The implementation realizes a number of stacked RNN layers using the ReservoirCell
    as core cell. All the reservoir layers share the same hyper-parameter values (i.e.,
    same number of recurrent neurons, spectral radius, etc..).
    """

    def __init__(
        self,
        input_size: int = 1,
        tot_units: int = 100,
        n_layers: int = 1,
        concat: bool = False,
        input_scaling: float = 1.0,
        inter_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_recurrent: int = 10,
        connectivity_input: int = 10,
        connectivity_inter: int = 10,
    ):
        """Initializes the DeepReservoir.

        Args:
            input_size (int): number of input units. Defaults to 1.
            tot_units (int): number of recurrent neurons in the reservoir. Defaults to 100.
            n_layers (int): number of stacked reservoir layers. Defaults to 1.
            concat (bool): if True, the output of each layer is concatenated to the
                previous ones. If False, only the output of the last layer is returned.
                Defaults to False.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            inter_scaling (float): max abs value of a weight in the input-reservoir
                connections between layers. Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_inter (int): number of outgoing connections from each
                reservoir unit to the next layer. Defaults to 10.
        """
        super().__init__()
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.batch_first = True  # DeepReservoir only supports batch_first

        # in case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed,
        # i.e., the number of trainable parameters does not depend on concat
        if concat:
            self.layers_units = np.int(tot_units / n_layers)
        else:
            self.layers_units = tot_units

        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            ReservoirLayer(
                input_size=input_size,
                units=self.layers_units + tot_units % n_layers,
                input_scaling=input_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                connectivity_input=connectivity_input_1,
                connectivity_recurrent=connectivity_recurrent,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concat=True
        last_h_size = self.layers_units + tot_units % n_layers
        for _ in range(n_layers - 1):
            reservoir_layers.append(
                ReservoirLayer(
                    input_size=last_h_size,
                    units=self.layers_units,
                    input_scaling=input_scaling_others,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_others,
                    connectivity_recurrent=connectivity_recurrent,
                )
            )
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, X: torch.Tensor):
        """Forward pass.

        Args:
            X (torch.Tensor): Input tensor, shaped as (batch, seq_len, n_inp).
        Returns:
            torch.Tensor: Output tensor, shaped as (batch, seq_len, n_out).
        """
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        # states_last is a list because different layers may have different size.

        for _, res_layer in enumerate(self.reservoir):
            [X, h_last] = res_layer(X)
            states.append(X)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        return states, states_last
