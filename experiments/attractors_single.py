import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import nolds
import os

from acds.archetypes.ron import RandomizedOscillatorsNetwork


def pca(all_states, pca_dim, out_dir, suffix_file=""):
    # perform PCA on all_states of shape (num_examples, n_hid)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim)
    pca_result = pca.fit_transform(all_states)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"Total variance explained by first 3 components: {np.sum(pca.explained_variance_ratio_)}")

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


def fractal_dim(data, seq_len, out_dir, corr=True, lyap=True):
    # compute correlation dimension for each principal component separately
    dims = []
    ls = []
    for i in range(data.shape[-1]):  # for each principal component
        avg_dim = 0.
        avg_l = 0.
        seq_count = 0.
        for j in tqdm(range(0, data.shape[0], seq_len), desc=f"Computing fractal dimension for PC{i}"):  # for each trajectory
            seq_count += 1
            dim = nolds.corr_dim(data[j:j+seq_len, i], emb_dim=10, lag=1) if corr else -1.
            l = nolds.lyap_r(data[j:j+seq_len, i]) if lyap else -1.
            # nolds may return a tuple, use the first element if so
            if isinstance(dim, tuple):
                avg_dim += dim[0]
            else:
                avg_dim += dim
            if isinstance(l, tuple):
                avg_l += l[0]
            else:
                avg_l += l
        avg_dim /= float(seq_count)
        avg_l /= float(seq_count)
        dims.append(avg_dim)
        ls.append(avg_l)

        print(f"Average correlation dimension over all trajectories for PC{i}: {avg_dim}")
        print(f"Average max Lyapunov exponent over all trajectories for PC{i}: {avg_l}")

    with open(os.path.join(out_dir, "fractal_dim.txt"), "w") as f:
        for i in range(len(dims)):
            f.write(f"PC{i}: {dims[i]};  ")
            f.write(f"Max Lyapunov PC{i}: {ls[i]}\n")

    return dims, ls


def main(args):
    seq_len = args.timesteps - args.washout

    # Prepare output directory 
    out_dir = os.path.join("results_single", f"rho_{args.rho}_nhid_{args.n_hid}_timesteps_{args.timesteps}")
    os.makedirs(out_dir, exist_ok=True)

    # Fix all random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the RandomizedOscillatorsNetwork instance
    ron = RandomizedOscillatorsNetwork(
        n_inp=args.n_inp,
        rho=args.rho,
        n_hid=args.n_hid,
        dt=args.dt,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device=args.device,
        input_scaling=0.0,  # No input, no bias
    )

    # Generate zero input for specified time steps and batch size
    zero_input = torch.zeros(args.batch_size, args.timesteps, args.n_inp)

    all_states = []

    for it in tqdm(range(args.n_init_states), desc="Computing trajectories"):
        # Generate random initial hidden states in [-1, 1]
        hs = (
            torch.rand(args.batch_size, args.n_hid) * 2 - 1,
            torch.rand(args.batch_size, args.n_hid) * 2 - 1
        )
        hs = (hs[0].to(args.device), hs[1].to(args.device))

        # Feed the model and collect all hy activations
        with torch.no_grad():
            hidden_states, _ = ron(zero_input, hs=hs)

        # remove washout
        hidden_states = hidden_states[:, args.washout:, :].cpu().numpy()

        all_states.append(hidden_states.squeeze(0))
    
    all_states = np.concatenate(all_states, axis=0)  # shape (n_init_states*timesteps, n_hid)
    np.save(os.path.join(out_dir, "all_states.npy"), all_states)
    
    pca_result = pca(all_states, args.pca_dim, out_dir)

    fractal_dim(pca_result, seq_len, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RandomizedOscillatorsNetwork with configurable parameters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_inp", type=int, default=1, help="Number of input units")
    parser.add_argument("--n_hid", type=int, default=10, help="Number of hidden units")
    parser.add_argument("--dt", type=float, default=1, help="Time step")
    parser.add_argument("--rho", type=float, default=0.4, help="Spectral radius")
    parser.add_argument("--gamma", type=float, default=1, help="Damping factor")
    parser.add_argument("--epsilon", type=float, default=1, help="Stiffness factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--timesteps", type=int, default=3000, help="Number of time steps")
    parser.add_argument("--washout", type=int, default=1000, help="Time steps to washout")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--n_init_states", type=int, default=1000, help="Number of initial states to generate")
    parser.add_argument("--pca_dim", type=int, default=2, help="Number of PCA dimensions")
    args = parser.parse_args()

    main(args)
