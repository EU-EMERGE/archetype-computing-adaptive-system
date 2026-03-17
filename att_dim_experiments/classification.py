import argparse
import torch
import numpy as np
import wandb
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from acds.archetypes import InterconnectionRON, DeepReservoir
from acds.networks import ArchetipesNetwork, random_matrix, full_matrix, cycle_matrix, deep_reservoir, star_matrix, local_connections
from aeon.datasets import load_classification
from acds.networks.utils import unstack_state
from acds.metrics import *
from acds.metrics import compute_lyapunov


cm_dict = {
    "random": random_matrix, 
    "full": full_matrix, 
    "cycle": cycle_matrix,
    "deep": deep_reservoir,
    "deep_no_skip": deep_reservoir,
    "star": star_matrix,
    "local": local_connections,
    "local_no_skip": local_connections,
    }



def get_connection_matrix(name: str, n_modules: int, p: float = 0.5, seed: Optional[int] = None) -> torch.Tensor:

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
        args['gamma'] = 1.0

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


    cm_type = args.get("connection_matrix", "cycle")
    connection_matrix = cm_dict[cm_type]
    # input connection mask
    input_mask = torch.ones((args['n_modules'],))
    if cm_type == "deep_no_skip" or cm_type == "local_no_skip":
         input_mask[1:] = 0 # only first module gets input

    if args.get("connection_matrix", "cycle") == "random":
        p = args.get("p", 0.2)
        connection_matrix = lambda n: random_matrix(n, p)
    network = ArchetipesNetwork(modules, connection_matrix(args['n_modules']), rho_m = args.get('rho_m', 1.0), input_mask=input_mask)
    network = DeepReservoir(
        input_size=n_input,
        tot_units=args['n_hid'],
        n_layers=1,
        leaky=args.get('alpha', 1.0),
        spectral_radius=args.get('rho', 1.0),
        input_scaling=args.get('input_scaling', 1.0),
    )
    return network


def get_data(args):
    train_dataset, train_target = load_classification(name=args['dataset'], extract_path='att_dim_experiments/data', split='train')
    train_dataset = np.permute_dims(train_dataset, [0, 2, 1])

    test_dataset, test_target = load_classification(name=args['dataset'], split="test", extract_path='att_dim_experiments/data')
    test_dataset = np.permute_dims(test_dataset, [0, 2, 1])
    # cast data from -1, 1 to 0, 1
    if np.unique(train_target).astype(int).tolist() == [-1, 1]:
       train_target = (train_target.astype(int) + 1) // 2
       test_target = (test_target.astype(int) + 1) // 2

    print(np.unique_counts(train_target), np.unique_counts(test_target))
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #n_samples, seq_len, n_features = train_dataset.shape
    #train_dataset = scaler.fit_transform(train_dataset.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    #n_samples, seq_len, n_features = test_dataset.shape
    #test_dataset = scaler.transform(test_dataset.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    

    

   # train_dataset, val_dataset, train_target, val_target = train_test_split(train_dataset, train_target, random_state=42, test_size=0.25, stratify=train_target)
    return (train_dataset, train_target.astype(int)), (test_dataset, test_target.astype(int))


def get_params_from_model(model: ArchetipesNetwork):
    state = unstack_state(model.archetipes_params, model.archetipes_buffers)
    return state


def compute_states(model:ArchetipesNetwork, data:torch.Tensor):
    model.eval()
    with torch.no_grad():
        states, fbs =  model.forward(data)
    return states, fbs


def train_readout(states:torch.Tensor, labels:torch.Tensor, ridge_alpha:float):
    n_modules, n_hid = states.shape[1], states.shape[-1]

    X = states[-1, :, :, 0, :].reshape(states.shape[2], n_modules * n_hid).numpy()  # (len_seq, n_mod, n_samples, 2, h_dim) -> (n_samples, n_mod * h_dim)
    y = labels.numpy()
    ridge = RidgeClassifierCV(cv=10, alphas=np.logspace(-6, 6, 13))
    ridge.fit(X, y)

    print("Best ridge alpha:", ridge.alpha_)


    return ridge


def evaluate(states:torch.Tensor, labels:torch.Tensor, readout:RidgeClassifierCV):
    n_modules, n_hid = states.shape[1], states.shape[-1]
    #X = states[-1, :, :, 0, :].reshape(states.shape[2], n_modules * n_hid).numpy()  # (len_seq, n_mod, n_samples, 2, h_dim) -> (n_samples, n_mod * h_dim)
    y = labels.numpy()
    predictions = readout.predict(X)
    print(X.max(), X.min())
    print(np.unique_counts(predictions))
    return readout.score(X, y) # acc
    

def run_classification():
    args=parse_args()
    args_dict = vars(args)

    with wandb.init(project=f"archetype-classification", config=args_dict):

        (train_data, train_labels), (test_data, test_labels) = get_data(args_dict)

        if not args.silent:
            print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
            print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

        # Get model
        n_input = train_data.shape[-1] if len(train_data.shape) > 1 else 1
        model = get_model(args_dict, n_input=n_input)
        params = get_params_from_model(model)
        if not args.silent:
            print(f"Model architecture: {model}")
        
            # Get reservoir states
        train_states, train_fbs = compute_states(model, torch.Tensor(train_data)) # (seq_len, n_modules, batch_size n_states, n_hid)
        test_states, test_fbs = compute_states(model, torch.Tensor(test_data))
        if not args.silent:
            print(f"Train states shape: {train_states.shape}")

            print(f"Test states shape: {test_states.shape}")

        readout = train_readout(train_states, torch.Tensor(train_labels), ridge_alpha=args.ridge_alpha)
        if not args.silent:
            print("Readout trained.")
            # Evaluate on validation and test sets
            print("Evaluating...")
            print("Train set:")
        train_score = evaluate(train_states, torch.Tensor(train_labels), readout)
        if not args.silent:
            print("Test set:")
        test_score = evaluate(test_states, torch.Tensor(test_labels), readout)


        train_states_np = np.transpose(train_states.numpy(), [2, 0, 1, 3, 4]) # (bs, seq_len, n_modules, 2, n_hid)
        train_fbs_np = np.transpose(train_fbs.numpy(), [2, 0, 1, 3])
        
        corr_dims = []
        part_ratios = []
        lyapunov_exps = []
        # for state, fb in zip(train_states_np, train_fbs_np):
        #     state = state[:, :, 0]
        #     corr_dims.append(compute_corr_dim(state, transient=0))
        #     part_ratios.append(compute_participation_ratio(state, transient=0))
        #     l = []
        #     for i, p in enumerate(params):
        #         l.append(compute_lyapunov(nl=2, W=p['h2h']. numpy().T, V=p['x2h'].numpy().T, b=p['bias'].numpy(), h_traj=state[:, i], u_traj=train_data[i], fb_traj=fb[:, i]))
        #     lyapunov_exps.append(l)
        # mles = np.mean(np.array(lyapunov_exps), axis=0)[:, 0]
        # eff_ranks = compute_effective_kernel_rank(train_states_np[:, :, :, 0])


        metrics = {
            "train_acc": train_score,
            #"val_acc": readout.best_score_,
            "test_acc": test_score,
            "att_dim/correlation_dimension_mean": np.mean(corr_dims),
            "att_dim/participation_ratio_mean": np.mean(part_ratios),
            # "att_dim/effective_rank_mean": np.mean(eff_ranks),
            # "lyapunov/max_lyap_exp": mles,
            # "lyapunov/mle_mean": mles.mean()
        }
        metrics.update(
        {"att_dim/correlation_dimension": corr_dims,
        "att_dim/participation_ratio": part_ratios,
        # "att_dim/effective_rank": eff_ranks,
        })


        wandb.log(metrics)







    if not args.silent:
        print(f"Training score: {train_score}")
        print(f"Test score: {test_score}")


def parse_args():
    parser = argparse.ArgumentParser(description="Archetype Network Classification Experiment")
    parser.add_argument("--dataset", type=str, default="ECG200", help="Name of the dataset to use from aeon library")
    
    # architecture parameters
    parser.add_argument("--n_modules", type=int, default=4, help="Number of archetype modules in the network")
    parser.add_argument("--n_hid", type=int, default=20, help="Number of hidden units in each archetype")
    parser.add_argument("--connection_matrix", type=str, default="cycle",
                        choices=["random", "full", "cycle", "deep", "star", "local", "deep_no_skip", "bidirectional"],
                        help="Type of connection matrix to use")
    
    # esn parameters
    parser.add_argument("--rho_m", type=float, default=0.9, help="Spectral radius scaling for inter-module connections")
    parser.add_argument("--input_scaling", type=float, default=1.0, help="Input scaling for the archetype network")
    parser.add_argument("--rho", type=float, default=1.0, help="Spectral radius for each archetype module")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha parameter equivalent to the leak rate for the archetype dynamics")

    # fixed rho params
    parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="Diffusive gamma parameter for the archetype dynamics")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step for the archetype dynamics")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for the archetype dynamics")  
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon parameter for the archetype dynamics")
    

    # training parameters
    parser.add_argument("--p", type=float, default=0.5, help="Connection probability for random matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regression regularization strength")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for reservoir computation. If -1, the reservoir is computed in one batch")
    parser.add_argument("--silent", action="store_true", help="If set, suppresses print statements")
    

    return parser.parse_args()

if __name__ == '__main__':
    run_classification()