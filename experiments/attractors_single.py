import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import os

from acds.archetypes.ron import RandomizedOscillatorsNetwork


def pca(all_states, pca_dim, out_dir, suffix_file=""):
    # perform PCA on all_states of shape (num_examples, n_hid)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim)
    pca_result = pca.fit_transform(all_states)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"Total variance explained by first {pca_dim} components: {np.sum(pca.explained_variance_ratio_)}")

    np.save(os.path.join(out_dir, f"pca_result{suffix_file}.npy"), pca_result)

    # plot pca_result in scatter plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    if pca_dim == 2:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    elif pca_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')  # type: ignore
    else:
        raise ValueError("pca_dim must be 2 or 3")
    plt.savefig(os.path.join(out_dir, f"pca{suffix_file}.png"))
    plt.close()

    return pca_result


def main(args):
    seq_len = args.timesteps - args.washout

    # Prepare output directory 
    out_dir = os.path.join("/scratch/a.cossu/results_single", f"rho_{args.rho}_nhid_{args.n_hid}_inp_scaling_{args.inp_scaling}_timesteps_{args.timesteps}")
    os.makedirs(out_dir, exist_ok=True)

    # Fix all random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the RandomizedOscillatorsNetwork instance
    ron = RandomizedOscillatorsNetwork(
        n_inp=1,
        rho=args.rho,
        n_hid=args.n_hid,
        dt=args.dt,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device=args.device,
        input_scaling=args.inp_scaling,
    )
    ron.bias = torch.nn.Parameter(torch.zeros(args.n_hid).to(args.device), requires_grad=False)

    # Generate zero input for specified time steps
    zero_input = torch.zeros(1, args.timesteps, 1)

    all_states = []
    all_inputs = []
    for it in tqdm(range(args.n_init_states), desc="Computing trajectories"):
        # Generate random initial hidden states in [-1, 1]
        hs = (
            torch.rand(1, args.n_hid) * 2 - 1,
            torch.rand(1, args.n_hid) * 2 - 1
        )
        hs = (hs[0].to(args.device), hs[1].to(args.device))

        # Feed the model and collect all hy activations
        with torch.no_grad():
            input_signal = zero_input + torch.randn_like(zero_input)
            hidden_states, _ = ron(zero_input, hs=hs)
        
        # remove washout        
        all_inputs.append(input_signal[:, args.washout:].squeeze(-1).cpu().numpy())
        hidden_states = hidden_states[:, args.washout:, :].cpu().numpy()

        all_states.append(hidden_states.squeeze(0))
    
    np.save(os.path.join(out_dir, "all_states.npy"), all_states)
    
    # add a trailing dimension of shape 1 to all inputs
    all_inputs = np.concatenate(all_inputs, axis=0)  # shape (n_init_states, timesteps)
    all_states = np.concatenate(all_states, axis=0)  # shape (n_init_states*timesteps, n_hid)
    
    pca_result = pca(all_states, args.pca_dim, out_dir)

    np.savetxt(os.path.join(out_dir, "W.csv"), ron.h2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "V.csv"), ron.x2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "b.csv"), ron.bias.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
    np.save(os.path.join(out_dir, "u_timeseries.npy"), all_inputs)


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
    parser.add_argument("--pca_dim", type=int, default=2, help="Number of PCA dimensions")
    args = parser.parse_args()

    main(args)
