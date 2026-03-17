import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import torch
import os
import numpy as np
import wandb
import pandas as pd
from typing import Optional
from sklearn.linear_model import Ridge
from acds.archetypes import InterconnectionRON
from acds.networks import ArchetipesNetwork, random_matrix, full_matrix, cycle_matrix, deep_reservoir, star_matrix, local_connections
from att_dim_experiments.data_utils import get_mg17, get_lorenz, get_narma10
from acds.networks.utils import unstack_state
from acds.metrics import compute_corr_dim, compute_participation_ratio, compute_effective_traj_rank, nrmse
from acds.metrics import compute_lyapunov


def get_connection_matrix(name: str, n_modules: int, p: float = 0.5, seed: Optional[int] = None) -> torch.Tensor:
    cm_dict = {"random": random_matrix, 
               "full": full_matrix, 
               "cycle": cycle_matrix,
               "deep": deep_reservoir,
               "star": star_matrix,
               "local": local_connections}
    if name not in cm_dict:
        raise ValueError(f"Connection matrix '{name}' not recognized. Available options are: {list(cm_dict.keys())}")
    if name == "random":
        return cm_dict[name](n_modules, p=p, seed=seed)
    else:
        return cm_dict[name](n_modules)
    

def get_model(args, n_input: int = 1):
    modules = []
    if args.get('alpha') is not None:
        alpha = args.get('alpha')
        args['dt'] = np.sqrt(alpha)
        args['epsilon'] = 1 / args['dt']

    for _ in range(args['n_modules']):
        modules.append(InterconnectionRON(
            n_inp=n_input,
            n_hid=args.get('n_hid'),
            dt=args.get('dt'),
            gamma=args.get('gamma'),
            epsilon=args.get('epsilon'),
            diffusive_gamma=args.get('diffusive_gamma'),
            rho=args.get('rho'),
            input_scaling=args.get('input_scaling'),
        ))

    cm_dict = {"random": random_matrix, 
               "full": full_matrix, 
               "cycle": cycle_matrix,
               "deep": deep_reservoir,
                "line": deep_reservoir,
                "star": star_matrix,
                "local": local_connections,
                "bidirectional": local_connections 
                }
    cm_type = args.get("connection_matrix", "cycle")
    connection_matrix = cm_dict[cm_type]
    # input connection mask
    input_mask = torch.ones((args['n_modules'],))
    # if cm_type == "bidirectional" or cm_type == "deep":
    #     input_mask[1:] = 0 # only first module gets input

    if args.get("connection_matrix", "cycle") == "random":
        p = args.get("p", 0.2)
        connection_matrix = lambda n: random_matrix(n, p)
    network = ArchetipesNetwork(modules, connection_matrix(args['n_modules']), rho_m = args.get('rho_m', 1.0), input_mask=input_mask)
    return network


def get_data(args):
    dataset_name = args['dataset']
    allowed_datasets = ["mg17", "lorenz", "narma10"]
    if dataset_name == "mg17":
        return get_mg17(train_len=5000, val_len=3000, test_len=2000, forecasting_delay=1)
    elif dataset_name == "lorenz":
        return get_lorenz(train_len=5000, val_len=3000, test_len=2000, forecasting_delay=1)
    elif dataset_name == "narma10":
        return get_narma10(train_len=5000, val_len=3000, test_len=2000)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Available options are: {allowed_datasets}")


def get_params_from_model(model: ArchetipesNetwork):
    state = unstack_state(model.archetipes_params, model.archetipes_buffers)
    return state



def compute_states(model: ArchetipesNetwork, data: torch.Tensor, initial_states:Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Get reservoir states for the given data."""
    model.eval()
    with torch.no_grad():
        states, fbs = model.forward(data, initial_states)
    return states, fbs


def train_readout(states: torch.Tensor, labels: torch.Tensor, transient: int, ridge_alpha: float) -> Ridge:
    """Train a Ridge regression readout on the reservoir states."""
    n_modules, n_hid = states.shape[1], states.shape[3]
    X = states[:, :, 0, :].reshape(states.shape[0], n_modules * n_hid).numpy()  # Use only the first state of each module
    X = X[transient:] # discard transient
    y = labels[transient:].numpy()
    #ridge = Ridge(alpha=0.000)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X, y)
    return ridge


def evaluate(states: torch.Tensor, labels: torch.Tensor, readout: Ridge) -> float:
    """Evaluate the readout on the given states and labels."""
    n_modules, n_hid = states.shape[1], states.shape[3]
    X = states[:, :, 0, :].reshape(states.shape[0], n_modules * n_hid).numpy()  # Use only the first state of each module
    y = labels.numpy()
    y_pred = readout.predict(X)
    # plot y and y_pred for debug
    score = nrmse(y[1000:], y_pred[1000:])  # align predictions with true labels
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="Archetype Network Classification Experiment")
    parser.add_argument("--dataset", type=str, default="mg17", help="Name of the dataset to use")

    # architecture parameters
    parser.add_argument("--n_modules", type=int, default=4, help="Number of archetype modules in the network")
    parser.add_argument("--n_hid", type=int, default=20, help="Number of hidden units in each archetype")
    parser.add_argument("--connection_matrix", type=str, default="cycle",
                        choices=["random", "full", "cycle", "deep", "star", "local"],
                        help="Type of connection matrix to use")
    
    # esn parameters
    parser.add_argument("--rho_m", type=float, default=0.9, help="Spectral radius scaling for inter-module connections")
    parser.add_argument("--input_scaling", type=float, default=1.0, help="Input scaling for the archetype network")
    parser.add_argument("--rho", type=float, default=1.0, help="Spectral radius for each archetype module")
    # fixed RON dynamics parameters
    parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="Diffusive gamma parameter for the archetype dynamics")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step for the archetype dynamics")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for the archetype dynamics")  
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon parameter for the archetype dynamics")
    # (dt, gamma, epsilon) = (1.0, 1.0, 1.0) are the standard values for non leaky ESN
    # (dt, gamma, epsilon) = (sqrt(alpha), 1, sqrt(1/alpha)) are the standard values for leaky ESN with leak rate alpha
    parser.add_argument("--alpha", type=float, default=None, help="Leak rate for leaky ESN. If set, overrides dt and epsilon accordingly.")
    # training parameters
    parser.add_argument("--p", type=float, default=0.5, help="Connection probability for random matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regression regularization strength")
    parser.add_argument("--silent", action="store_true", help="If set, suppresses print statements")
    return parser.parse_args()


def run_forecasting():
    args = parse_args()
    args_dict = vars(args)

    # Initialize wandb
    wandb.init(project=f"archetype-forecasting-{args.dataset}", config=args_dict)

    # Get data
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = get_data(args_dict)
    if not args.silent:
        print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}, Validation labels shape: {val_labels.shape}")
        print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    # Get model
    n_input = train_data.shape[1] if len(train_data.shape) > 1 else 1
    model = get_model(args_dict, n_input=n_input)
    params = get_params_from_model(model)
    if not args.silent:
        print(f"Model architecture: {model}")

    # Get reservoir states
    train_states, train_fbs = compute_states(model, torch.Tensor(train_data)) # (seq_len, n_modules, n_states, n_hid)
    val_states, val_fbs = compute_states(model, torch.Tensor(val_data), initial_states=train_states[-1])
    test_states, test_fbs = compute_states(model, torch.Tensor(test_data), initial_states=val_states[-1])
    if not args.silent:
        print(f"Train states shape: {train_states.shape}")

    # Train readout
    ridge_alpha = args.ridge_alpha
    readout = train_readout(train_states, torch.Tensor(train_labels), transient=100, ridge_alpha=ridge_alpha)
    if not args.silent:
        print("Readout trained.")
    # Evaluate on validation and test sets
    train_score = evaluate(train_states[1000:], torch.Tensor(train_labels[1000:]), readout)
    val_score = evaluate(val_states, torch.Tensor(val_labels), readout)
    test_score = evaluate(test_states, torch.Tensor(test_labels), readout)
    
    # compute metrics on val_states
    val_states_np = val_states.numpy()[:, :, 0] # (seq_len, n_modules, n_hid)
    val_fbs_np = val_fbs.numpy()

    corr_dim = compute_corr_dim(val_states_np, transient=1000)
    part_ratio = compute_participation_ratio(val_states_np, transient=1000)
    eff_rank = compute_effective_traj_rank(val_states_np, transient=1000)
    lyapunov_exps = []
    for i, p in enumerate(params):
        lyap = compute_lyapunov(nl=2, W=p['h2h']. numpy().T, V=p['x2h'].numpy().T, b=p['bias'].numpy(), h_traj=val_states_np[:, i], u_traj=val_data, fb_traj=val_fbs_np[:, i])
        lyapunov_exps.append(lyap)
    if not args.silent:
        print(f"Correlation Dimension per module: {corr_dim}")
        print(f"Participation Ratio per module: {part_ratio}")
        print(f"Effective Rank per module: {eff_rank}")
        print(f"Maximum Lyapunov Exponent per module: {[l for l in lyapunov_exps]}")
        print(f"Train NRMSE score: {train_score:.4f}")
        print(f"Validation NRMSE score: {val_score:.4f}")
        print(f"Test NRMSE score: {test_score:.4f}")

    metrics = {
        "train_nrmse": train_score,
        "val_nrmse": val_score,
        "test_nrmse": test_score,
        "att_dim/correlation_dimension_mean": np.mean(corr_dim),
        "att_dim/participation_ratio_mean": np.mean(part_ratio),
        "att_dim/effective_rank_mean": np.mean(eff_rank),
    }
    
    # Log metrics to wandb
    wandb.log(metrics)

    metrics.update(
        {"att_dim/correlation_dimension": corr_dim,
        "att_dim/participation_ratio": part_ratio,
        "att_dim/effective_rank": eff_rank,
        })

    # log per-module metrics as tables
    cd_table = wandb.Table(columns=["module", "correlation_dimension"])
    pr_table = wandb.Table(columns=["module", "participation_ratio"])
    er_table = wandb.Table(columns=["module", "effective_rank"])
    for i in range(len(corr_dim)):
        cd_table.add_data(i, corr_dim[i])
        pr_table.add_data(i, part_ratio[i])
        er_table.add_data(i, eff_rank[i])
    wandb.log({
        "att_dim/correlation_dimension_per_module": cd_table,
        "att_dim/participation_ratio_per_module": pr_table,
        "att_dim/effective_rank_per_module": er_table,
    })
    lyapunov_table = wandb.Table(columns=["module", "lyapunov exponents"])
    mle_table = wandb.Table(columns=["module", "mle"])
    for i in range(len(corr_dim)):
        lyapunov_table.add_data(i, lyapunov_exps[i])
        mle_table.add_data(i, lyapunov_exps[i][0]) # maximum
    wandb.log({
        "lyapunov/lyapunov_exponents": lyapunov_table,
        "lyapunov/mle": mle_table
    })
    metrics.update(
        {"att_dim/lyapunov": lyapunov_exps}
    )
    metrics.update(vars(args))
    df = pd.DataFrame([metrics])
    file_exists = os.path.isfile("trajectories/metrics.csv")
        
    df.to_csv("trajectories/metrics.csv", mode="a" if file_exists else "w", header=True, index=False)
    wandb.finish()



if __name__ == "__main__":
    run_forecasting()