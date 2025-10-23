import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from experiments.attractors_single import pca
from acds.archetypes.ron import RandomizedOscillatorsNetwork
from collections import defaultdict


def plot_combined_pca(pca_results, out_dir, labels=None):
    """
    Plot multiple PCA results in a single scatter plot with different markers.

    Args:
        pca_results: list of np.ndarray, each of shape (N, 2) or (N, 3)
        out_path: path to save the figure (e.g., 'combined_pca.png')
        labels: optional list of labels for the legend
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not pca_results:
        raise ValueError("pca_results list is empty")

    dim = pca_results[0].shape[1]
    assert dim in (2, 3), "PCA results must be 2D or 3D"

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    colors = [plt.get_cmap('tab10')(i) for i in range(10)]

    fig = plt.figure(figsize=(8, 6))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    for i, data in enumerate(pca_results):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f"Set {i+1}"
        if dim == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker, color=color, label=label, alpha=0.7)
        else:
            ax.scatter(data[:, 0], data[:, 1], marker=marker, color=color, label=label, alpha=0.7)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if dim == 3:
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel("PC3")  # type: ignore
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_combined.png"))
    plt.close()


def main(args):
    # Prepare output directory
    out_dir = os.path.join(
        "/scratch/a.cossu/results_collective",
        f"mod{args.n_modules}_rho_{args.rho}_nhid_{args.n_hid}_timesteps_{args.timesteps}_inpscaling_{args.inp_scaling}{args.suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Fix all random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    models = []
    for _ in range(args.n_modules):
        ron = RandomizedOscillatorsNetwork(
            n_inp=args.n_hid,
            n_hid=args.n_hid,
            dt=args.dt,
            rho=args.rho,
            gamma=args.gamma,
            epsilon=args.epsilon,
            device=args.device,
            input_scaling=args.inp_scaling
        )
        ron.bias = torch.nn.Parameter(torch.zeros(args.n_hid).to(args.device), requires_grad=False)
        models.append(ron)

    all_states = defaultdict(list)
    input_signals = {i: [] for i in range(args.n_modules)}
    for it in tqdm(range(args.n_init_states), desc="Computing trajectories"):
        # Random initial hidden states for both networks in [-1, 1]
        hs = []
        for _ in range(args.n_modules):
            h = (
                torch.rand(1, args.n_hid, device=args.device) * 2 - 1,
                torch.rand(1, args.n_hid, device=args.device) * 2 - 1
            )
            hs.append(h)

        states = defaultdict(list)
        inputs = {i: [] for i in range(args.n_modules)}
        with torch.no_grad():
            for t in range(args.timesteps):
                for i in range(args.n_modules):
                    input_idx = (i - 1) % args.n_modules  # Ring topology
                    input_signal = hs[input_idx][0] + torch.randn_like(hs[input_idx][0]) # noise mean 0 variance 1
                    hy, hz = models[i].cell(input_signal, hs[i][0], hs[i][1])
                    hs[i] = (hy, hz)
                    states[i].append(hy)
                    inputs[i].append(input_signal)
        
        for i in range(args.n_modules):
            traj_inp = torch.cat(inputs[i][args.washout:], dim=0)  # shape (timesteps - washout, n_hid)
            input_signals[i].append(traj_inp)

        hidden_states = {}
        for i in range(args.n_modules):
            hidden_states[i] = torch.stack(states[i], dim=1)[:, args.washout:, :].cpu().numpy().squeeze(0)
            all_states[i].append(hidden_states[i])

    torch.save(input_signals, os.path.join(out_dir, f"input_signals.pt"))

    for i in range(args.n_modules):
        np.save(os.path.join(out_dir, f"all_states{i}.npy"), all_states[i])

    pca_results = []
    for i in range(args.n_modules):
        pca_result = pca(np.concatenate(all_states[i], axis=0), args.pca_dim, out_dir, suffix_file=f"_{i}")
        pca_results.append(pca_result)
    plot_combined_pca(pca_results, out_dir, labels=[f"{i}" for i in range(args.n_modules)])

    for i, ron in enumerate(models):
        np.savetxt(os.path.join(out_dir, f"W_{i}.csv"), ron.h2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
        np.savetxt(os.path.join(out_dir, f"V_{i}.csv"), ron.x2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
        np.savetxt(os.path.join(out_dir, f"b_{i}.csv"), ron.bias.detach().cpu().numpy(), delimiter=',', fmt="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RandomizedOscillatorsNetwork with configurable parameters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_hid", type=int, default=10, help="Number of hidden units")
    parser.add_argument("--dt", type=float, default=1, help="Time step")
    parser.add_argument("--rho", type=float, default=0.4, help="Spectral radius")
    parser.add_argument("--inp_scaling", type=float, default=0.1, help="Input scaling")
    parser.add_argument("--gamma", type=float, default=1, help="Damping factor")
    parser.add_argument("--epsilon", type=float, default=1, help="Stiffness factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--timesteps", type=int, default=3000, help="Number of time steps")
    parser.add_argument("--washout", type=int, default=1000, help="Time steps to washout")
    parser.add_argument("--n_init_states", type=int, default=1000, help="Number of initial states to generate")
    parser.add_argument("--n_modules", type=int, default=2, help="Number of modules to use")
    parser.add_argument("--pca_dim", type=int, default=2, help="Number of PCA dimensions")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output files")
    args = parser.parse_args()

    main(args)
