import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import plotly.tools as tls
from acds.archetypes.utils import count_parameters
from collections import defaultdict
import os
import tensorboard

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork
)
parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--resultroot", type=str)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--delay", type=int, default=200)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--deepron", action="store_true")

parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--n_hid", type=int, default=100)
parser.add_argument("--dt", type=float, default=0.0075)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--gamma_range", type=float, default=0.5)
parser.add_argument("--epsilon_range", type=float, default=1)
parser.add_argument("--rho", type=float, default=0.99)
parser.add_argument("--inp_scaling", type=float, default=1)
parser.add_argument("--leaky", type=float, default=1.0, help="ESN spectral radius")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers of ESN")
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)
parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric", "orthogonal"], help="Topology of the hidden-to-hidden matrix")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--resultsuffix", type=str, default="")

args = parser.parse_args()

# setup gamma, epsilon and their range for optuna then calculate the final values and return as trial suggest
def gamma_r(trial):

    gamma = trial.suggest_float("gamma", 0.1, 0.5, 2.0, 3)
    gamma_range = trial.suggest_float("gamma_range", 0.1, 0.5)

    return (gamma - gamma_range / 2.0, gamma + gamma_range / 2.0) 

def epsilon_r(trial):
    
    epsilon = trial.suggest_float("epsilon", 0.1, 0.5, 3.0)
    epsilon_range = trial.suggest_float("epsilon_range", 0.1, 0.5)
    
    return (epsilon - epsilon_range / 2.0, epsilon + epsilon_range / 2.0)

def evaluate(output, target):
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1])**2

# setup ray for parameter tuning with optuna by Bayesian optimization



def train_memory_capacity(config):
    
    # Initialize model with config parameters
    if args.esn:
        model = DeepReservoir(
            input_size=1,
            tot_units=args.n_hid,
            spectral_radius=config["rho"],
            n_layers=int(config["n_layers"]),
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            leaky=args.leaky,
            concat=True,
            connectivity_input=int(args.n_hid / config["n_layers"]),
            connectivity_inter=int(args.n_hid / config["n_layers"]),
            connectivity_recurrent=int(args.n_hid / config["n_layers"]),
        )
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=1,
            n_layers=int(config["n_layers"]),
            total_units=args.n_hid,
            dt=config["dt"],
            gamma=config["gamma"],
            epsilon=config["epsilon"],
            input_scaling=args.inp_scaling,
            inter_scaling=args.inp_scaling,
            reservoir_scaler=args.inp_scaling,
            connectivity_input=int(args.n_hid / config["n_layers"]),
            connectivity_inter=int(args.n_hid / config["n_layers"]),
            rho=config["rho"],
        )
    
    # Add other model types as needed
    # Generate input signal

    T = 5000  # Total timesteps
    train_steps = 4000
    valid_steps = 1000
    washout = 100
    u = np.random.uniform(-0.8, 0.8, (T+args.delay, 1))
    u  = u.astype(np.float64)
    
    
    # Forward pass
    hidden_states = []
    with torch.no_grad():
        hidden_states = model(torch.tensor(u[:-args.delay]).to(device="cpu").reshape(1, -1, 1))[0].cpu().numpy()
        hidden_states = hidden_states.reshape(-1, args.n_hid)
        
    hidden_states = np.array(hidden_states).astype(np.float64)
    
    # Calculate Memory Capacity
    mc = 0.0
    max_lag = 2 * 100
    for k in tqdm(range(max_lag + 1)):
        # Create delayed targets
        states = hidden_states[k:T, :]
        targets = u[: T - k, 0]
   
        # split
        split_idx_train = train_steps - k
        split_idx_valid = split_idx_train + valid_steps
        
        y_train, y_valid = (targets[:split_idx_train], targets[split_idx_train:split_idx_valid])
        X_train, X_valid = (states[:split_idx_train, :], states[split_idx_train:split_idx_valid, :])
        
        # remove washout
        y_train, y_valid = y_train[washout:], y_valid[washout:]
        X_train, X_valid = X_train[washout:], X_valid[washout:] 
        
        # Scaler    
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
   
        # Train Ridge regression
        ridge = Ridge(alpha=1e-6, max_iter=1000)
        ridge.fit(X_train, y_train)
         
        # Predict and calculate correlation
        preds = ridge.predict(X_valid)
        r2 = evaluate(preds, y_valid)
        mc += r2

    # Report metric to Ray Tune
    train.report({"mc": mc})

def run_hyperparameter_search():
    # Bayesian optimization search 
    
    search_space = {
        "gamma": tune.uniform(1, 3),
        "epsilon": tune.uniform(1, 3.0),
        "dt": tune.uniform(0.5, 1.0),
        "rho": tune.uniform(0.99, 0.99),
        "n_layers": tune.uniform(args.n_layers, args.n_layers),
    }

    # Configure Bayesian optimization
    bayesopt = BayesOptSearch(
        metric="mc",
        mode="max",
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
    )
    
    bayesopt = ConcurrencyLimiter(bayesopt, max_concurrent=4)

    tuner = tune.Tuner(
        train_memory_capacity,
        tune_config=tune.TuneConfig(
            search_alg=bayesopt,
            num_samples=50,  # Number of trials
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="mc", mode="max")
    print(f"Best MC: {best_result.metrics['mc']}")
    print(f"Best config: {best_result.config}")

if __name__ == "__main__":
    
    
    # Run hyperparameter search if specified
    if args.cpu:
        # check how many cpus are available and use them all
        print("Using all available cpus: ", os.cpu_count())
        ray.init(num_cpus=os.cpu_count())
    else:    
        ray.init(num_cpus=8)

    run_hyperparameter_search()

    ray.shutdown()
    
