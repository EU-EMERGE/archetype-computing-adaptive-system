import argparse
import importlib
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from acds.archetypes import DeepReservoir, UnicycleReservoir
except (ModuleNotFoundError, TypeError):
    for module_name in list(sys.modules):
        if module_name == "acds" or module_name.startswith("acds."):
            del sys.modules[module_name]

    acds_pkg = types.ModuleType("acds")
    acds_pkg.__path__ = [str(REPO_ROOT / "acds")]
    archetypes_pkg = types.ModuleType("acds.archetypes")
    archetypes_pkg.__path__ = [str(REPO_ROOT / "acds" / "archetypes")]
    sys.modules["acds"] = acds_pkg
    sys.modules["acds.archetypes"] = archetypes_pkg

    DeepReservoir = importlib.import_module("acds.archetypes.esn").DeepReservoir
    UnicycleReservoir = importlib.import_module("acds.archetypes.run").UnicycleReservoir


def parse_delays(spec):
    delays = []
    for item in spec.split(","):
        if not item.strip():
            continue
        parts = [int(x) for x in item.split(":")]
        if len(parts) == 1:
            delays.append(parts[0])
        elif len(parts) in (2, 3):
            start, stop = parts[:2]
            step = parts[2] if len(parts) == 3 else 1
            delays.extend(range(start, stop + 1, step))
        else:
            raise ValueError(f"Invalid delay spec: {item}")
    delays = sorted(set(delays))
    if not delays or min(delays) < 0:
        raise ValueError("Delays must be non-negative.")
    return delays


def make_distractor(delay, args, rng):
    if delay == 0:
        return np.empty(0, dtype=np.float32)
    if args.distractor_scaling == 0:
        raise ValueError("Positive delays require a nonzero distractor_scaling.")

    distractor = rng.normal(0.0, args.distractor_scaling, delay).astype(np.float32)
    while np.any(distractor == 0):
        distractor = rng.normal(0.0, args.distractor_scaling, delay).astype(np.float32)
    return distractor


def stimulus_values(args):
    return np.array(
        [-args.stimulus_scaling, args.stimulus_scaling],
        dtype=np.float32,
    )


def make_sequence(delay, args, rng):
    x = np.zeros((delay + 2, 1), dtype=np.float32)
    values = stimulus_values(args)

    sample = int(rng.integers(0, 2))
    probe = int(rng.integers(0, 2))
    x[0, 0] = values[sample]
    x[1 : delay + 1, 0] = make_distractor(delay, args, rng)
    x[delay + 1, 0] = values[probe]

    return torch.from_numpy(x)


def make_unicycle_input_map(args):
    return (torch.rand(1, args.hidden_size) * 2 - 1) * args.input_scaling


def build_reservoir(args):
    if args.model == "deep_reservoir":
        return DeepReservoir(
            input_size=1,
            tot_units=args.hidden_size,
            connectivity_recurrent=args.hidden_size,
            connectivity_input=args.hidden_size,
            input_scaling=args.input_scaling,
            spectral_radius=args.rho,
            leaky=args.leaky,
        )

    return UnicycleReservoir(
        n_inp=1,
        n_units=args.hidden_size,
        dt=args.unicycle_dt,
        n_out=1,
        lin_input_map=make_unicycle_input_map(args),
        ang_input_map=make_unicycle_input_map(args),
        n_connections=args.hidden_size,
        n_connections_anchor=args.hidden_size,
        n_connections_ang=args.hidden_size,
        n_connections_anchor_ang=args.hidden_size,
    )


def initial_state(args):
    if args.model == "deep_reservoir":
        return torch.randn(1, args.hidden_size)

    x = torch.randn(1, args.hidden_size)
    z = torch.randn(1, args.hidden_size)
    theta = torch.randn(1, args.hidden_size) * (4 * math.pi) - (2 * math.pi)
    s = torch.randn(1, args.hidden_size)
    omega = torch.randn(1, args.hidden_size)
    return x, z, theta, s, omega


def clone_state(args, state):
    if args.model == "deep_reservoir":
        return state.clone()
    return tuple(component.clone() for component in state)


def perturb_state(args, state, scale=1e-6):
    if args.model == "deep_reservoir":
        return state + torch.randn_like(state) * scale
    return tuple(component + torch.randn_like(component) * scale for component in state)


def step_reservoir(reservoir, args, input_t, state):
    if args.model == "deep_reservoir":
        h, _ = reservoir.reservoir[0].net(input_t.unsqueeze(0), state)
        return h

    x, z, theta, s, omega = state
    input_t = input_t.unsqueeze(0)
    linear_input = (input_t + reservoir.inp_bias) @ reservoir.lin_input_map
    angular_input = input_t @ reservoir.ang_input_map
    return reservoir.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)


def state_vector(args, state):
    if args.model == "deep_reservoir":
        return state
    return torch.hstack(state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local perturbation metric for the variable-delay match-to-sample task"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deep_reservoir",
        choices=("deep_reservoir", "unicycle"),
        help="Reservoir model to evaluate",
    )
    parser.add_argument("--hidden_size", type=int, default=10, help="Hidden size of the reservoir")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to average over")
    parser.add_argument(
        "--delays",
        type=str,
        default="1,2,5,10,20,30,50,75,100,150,200",
        help="Comma/range delay specification, e.g. 1,2,5 or 1:20:2",
    )
    parser.add_argument("--distractor_scaling", type=float, default=0.01)
    parser.add_argument("--stimulus_scaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input_scaling", type=float, default=1.0, help="Input scaling of the reservoir")
    parser.add_argument("--rho", type=float, default=0.9, help="Spectral radius of the reservoir")
    parser.add_argument("--leaky", type=float, default=1.0, help="Leaky parameter of the reservoir")
    parser.add_argument("--unicycle_dt", type=float, default=0.001, help="Integration time step for the Unicycle reservoir")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.hidden_size > 0
    assert args.num_samples > 0
    assert args.stimulus_scaling != 0

    if args.seed is not None:
        torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    delays = parse_delays(args.delays)
    reservoir = build_reservoir(args)

    dnorm = {}
    dcs = defaultdict(float)
    d0s = defaultdict(float)

    with torch.no_grad():
        for delay in delays:
            for _ in range(args.num_samples):
                input_sequence = make_sequence(delay, args, rng)

                h1 = initial_state(args)
                h2 = perturb_state(args, clone_state(args, h1))
                for t in range(delay + 2):
                    h1 = step_reservoir(reservoir, args, input_sequence[t], h1)
                    h2 = step_reservoir(reservoir, args, input_sequence[t], h2)
                    if t == 0:
                        h10 = clone_state(args, h1)
                        h20 = clone_state(args, h2)

                d_c = torch.norm(state_vector(args, h2) - state_vector(args, h1))
                d_0 = torch.norm(state_vector(args, h20) - state_vector(args, h10))
                dcs[delay] += d_c.item()
                d0s[delay] += d_0.item()

            dcs[delay] /= args.num_samples
            d0s[delay] /= args.num_samples
            dnorm[delay] = dcs[delay] / d0s[delay]
            print(f"D_context(T={delay}): {dcs[delay]}")
            print(f"D_context(0): {d0s[delay]}")
            print(f"Normalized D_context(T={delay}): {dnorm[delay]}")

    print(dcs)


if __name__ == "__main__":
    main()
