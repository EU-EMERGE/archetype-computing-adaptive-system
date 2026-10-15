import torch
from torch import nn
from acds.archetypes.utils import sparse_tensor_init


def skewsymmetric(units: int, recur_scaling: float) -> torch.FloatTensor:
    """ Generate a skewsymmetric matrix.
    """
    W = recur_scaling * ( 2 * torch.rand(units, units) - 1) # uniform in (-recur_scaling, recur_scaling)
    W = W - W.T
    return W


class EulerReservoirCell(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1.,
                 connectivity_input=10, bias_bool=True, nonlin='tanh',
                 bias_scaling=None, epsilon=1.0, gamma=1.0,
                 recur_scaling=1.0):
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.connectivity_input = connectivity_input

        self.kernel = sparse_tensor_init(input_size, self.units,
                                         self.connectivity_input) * self.input_scaling
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        W = skewsymmetric(units, recur_scaling)
        self.recurrent_kernel = W - gamma * torch.eye(units)
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)

        if bias_bool:
            if bias_scaling is None:
                self.bias_scaling = self.input_scaling
            else:
                self.bias_scaling = bias_scaling
            # uniform init in [-1, +1] times bias_scaling
            self.bias = (2 * torch.rand(self.units) - 1) * self.bias_scaling
            self.bias = nn.Parameter(self.bias, requires_grad=False)
        else:
            # zero bias
            self.bias = torch.zeros(self.units)
            self.bias = nn.Parameter(self.bias, requires_grad=False)

        self.epsilon = torch.tensor([epsilon])
        self.epsilon = nn.Parameter(self.epsilon, requires_grad=False)

        self.nonlin = nonlin

    def forward(self, xt, h_prev):
        input_part = torch.mm(xt, self.kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)

        if self.nonlin == 'identity':
            output = input_part + self.bias + state_part
        elif self.nonlin == 'tanh':
            output = torch.tanh(input_part + self.bias + state_part)
        else:
            raise ValueError("Invalid nonlinearity <<" + self.nonlin + ">>. Only tanh and identity allowed.")

        out = h_prev + output * self.epsilon
        return out, out


class EuESNLayer(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1.,
                 connectivity_input=10, bias_bool=True, nonlin='tanh',
                 bias_scaling=None, epsilon=1.0, gamma=1.0,
                 recur_scaling=1.0):
        super().__init__()
        self.net = EulerReservoirCell(
            input_size, units, input_scaling, connectivity_input, bias_bool,
            nonlin, bias_scaling, epsilon, gamma, recur_scaling
        )

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.net.units)

    def forward(self, x, h_prev=None):

        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0]).to(x.device)

        hs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            _, h_prev = self.net(xt, h_prev)
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev

class DeepEuESN(torch.nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                input_scaling=1, inter_scaling=1,
                connectivity_input=10,
                connectivity_inter=10,
                bias_bool=True,
                nonlin='tanh',
                bias_scaling=None,
                epsilon=1.0,
                gamma=1.0,
                recur_scaling=1.0):
        super().__init__()
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.batch_first = True  # DeepEuESN only supports batch_first

        # in case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed,
        # i.e., the number of trainable parameters does not depend on concat
        if concat:
            self.layers_units = int(tot_units / n_layers)
        else:
            self.layers_units = tot_units

        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            EuESNLayer(
                input_size=input_size,
                units=self.layers_units + tot_units % n_layers,
                input_scaling=input_scaling,
                connectivity_input=connectivity_input_1,
                bias_bool=bias_bool,
                nonlin=nonlin,
                bias_scaling=bias_scaling,
                epsilon=epsilon,
                gamma=gamma,
                recur_scaling=recur_scaling)
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concat=True
        last_h_size = self.layers_units + tot_units % n_layers
        for _ in range(n_layers - 1):
            reservoir_layers.append(
                EuESNLayer(
                    input_size=last_h_size,
                    units=self.layers_units,
                    input_scaling=input_scaling_others,
                    connectivity_input=connectivity_input_others,
                    bias_bool=bias_bool,
                    nonlin=nonlin,
                    bias_scaling=bias_scaling,
                    epsilon=epsilon,
                    gamma=gamma,
                    recur_scaling=recur_scaling,
                )
            )
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, X):
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        # states_last is a list because different layers may have different size.

        for res_layer in self.reservoir:
            [X, h_last] = res_layer(X)
            states.append(X)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        return states, states_last
