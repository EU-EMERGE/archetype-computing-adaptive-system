import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acds.archetypes import DeepReservoir, UnicycleReservoir


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default='deep_reservoir',
    choices=('deep_reservoir', 'unicycle'),
    help='Reservoir model to evaluate',
)
parser.add_argument('--input_size', type=int, default=1, help='Input size of the reservoir')
parser.add_argument('--context_length', type=int, default=1, help='Context length')
parser.add_argument('--hidden_size', type=int, default=10, help='Hidden size of the reservoir')
parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to average over')
parser.add_argument('--input_scaling', type=float, default=1.0, help='Input scaling of the reservoir')
parser.add_argument('--rho', type=float, default=0.9, help='Spectral radius of the reservoir')
parser.add_argument('--leaky', type=float, default=1.0, help='Leaky parameter of the reservoir')
parser.add_argument('--unicycle_dt', type=float, default=0.001, help='Integration time step for the Unicycle reservoir')
args = parser.parse_args()


def make_unicycle_input_map():
    return (torch.rand(args.input_size, args.hidden_size) * 2 - 1) * args.input_scaling


def build_reservoir():
    if args.model == 'deep_reservoir':
        return DeepReservoir(input_size=args.input_size,
                             tot_units=args.hidden_size,
                             connectivity_recurrent=args.hidden_size,
                             connectivity_input=args.hidden_size,
                             input_scaling=args.input_scaling,
                             spectral_radius=args.rho,
                             leaky=args.leaky)

    return UnicycleReservoir(n_inp=args.input_size,
                             n_units=args.hidden_size,
                             dt=args.unicycle_dt,
                             n_out=args.input_size,
                             lin_input_map=make_unicycle_input_map(),
                             ang_input_map=make_unicycle_input_map(),
                             n_connections=args.hidden_size,
                             n_connections_anchor=args.hidden_size,
                             n_connections_ang=args.hidden_size,
                             n_connections_anchor_ang=args.hidden_size)


def initial_state():
    if args.model == 'deep_reservoir':
        return torch.zeros(1, args.hidden_size)

    x = torch.rand(1, args.hidden_size)
    z = torch.rand(1, args.hidden_size)
    theta = torch.rand(1, args.hidden_size) * (4 * math.pi) - (2 * math.pi)
    s = torch.zeros(1, args.hidden_size)
    omega = torch.zeros(1, args.hidden_size)
    return x, z, theta, s, omega


def clone_state(state):
    if args.model == 'deep_reservoir':
        return state.clone()
    return tuple(component.clone() for component in state)


def step_reservoir(input_t, state):
    if args.model == 'deep_reservoir':
        h, _ = reservoir.reservoir[0].net(input_t.unsqueeze(0), state)
        return h

    x, z, theta, s, omega = state
    input_t = input_t.unsqueeze(0)
    linear_input = (input_t + reservoir.inp_bias) @ reservoir.lin_input_map
    angular_input = input_t @ reservoir.ang_input_map
    return reservoir.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)


def state_vector(state):
    if args.model == 'deep_reservoir':
        return state
    return torch.hstack(state)


reservoir = build_reservoir()

Ts = [1, 2, 4, 8, 16, 32, 64, 128]

dnorm = {}
dcs = defaultdict(float)
d0s = defaultdict(float)
for ts in Ts:
    for n in range(args.num_samples):
        input_sequence1 = torch.randn(ts + args.context_length, args.input_size)
        input_sequence2 = input_sequence1.clone()
        input_sequence2[:args.context_length] = torch.randn(args.context_length, args.input_size)

        h1 = initial_state()
        h2 = clone_state(h1)
        for t in range(ts + args.context_length):
            h1 = step_reservoir(input_sequence1[t], h1)
            h2 = step_reservoir(input_sequence2[t], h2)
            if t == args.context_length - 1:
                h10 = clone_state(h1)
                h20 = clone_state(h2)

        d_c = torch.norm(state_vector(h2) - state_vector(h1))
        d_0 = torch.norm(state_vector(h20) - state_vector(h10))
        dcs[ts] += d_c
        d0s[ts] += d_0
    dcs[ts] /= args.num_samples
    d0s[ts] /= args.num_samples
    dnorm[ts] = dcs[ts] / d0s[ts]
    print(f"D_context(T={ts}): {dcs[ts]}")
    print(f"D_context(0): {d0s[ts]}")
    print(f"Normalized D_context(T={ts}): {dnorm[ts]}")

print(dnorm)
