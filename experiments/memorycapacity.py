import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import warnings
import wandb
import matplotlib.pyplot as plt
import plotly.tools as tls
import logging
from collections import defaultdict



# import plotly
import plotly.express as px

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    DeepRandomizedOscillatorsNetwork
)
# Import memory capacity
from acds.benchmarks import get_memory_capacity

parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--resultroot", type=str)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--delay", type=int, default=100)
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
parser.add_argument("--n_hid_layers", type=str, default="256, 256", help="hidden size of recurrent net")
parser.add_argument("--n_layers", type=int, default=1, help="Number of layers of ESN")
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)

parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric"], help="Topology of the hidden-to-hidden matrix")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)

parser.add_argument("--resultsuffix", type=str, default="")

args = parser.parse_args()

# make sure that n_hid_layers is a list of integers
args.n_hid_layers = [int(x) for x in args.n_hid_layers.split(",")]

# --- Wandb logging ---
if args.wandb == True:
    wandb.init(project="deep-ron-thesis",
            config={"architecture": "DeepRON" if args.deepron else "RON" if args.ron else "ESN",
                    "trials": args.trials, 
                    "n_hid": args.n_hid,
                    "delay": args.delay,
                    #"num_layers": len(args.n_hid_layers),
                    "gamma": args.gamma,
                    "diffusive_gamma": args.diffusive_gamma,
                    "epsilon": args.epsilon,
                    "rho": args.rho,
                    "inp_scaling": args.inp_scaling,
                    "leaky": args.leaky,
                    "sparsity": args.sparsity,
                    "topology": args.topology,
                    "use_test": args.use_test,
                    "resultsuffix": args.resultsuffix
                    }
            )
else:
    warnings.warn("Wandb is not enabled. No logging will be done.")
    # disable wandb logging
    wandb.disabled = True

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()

n_inp = 1
n_out = 1
washout = 100
# TODO this should be set automatically to double of units of the model
delay = args.delay

def square_correlation(output, target):
    return (np.corrcoef(output.flatten(), target.flatten())[0, 1])**2

# set custom criterion eval to square correlation
def criterion_eval(output, target):
    return square_correlation(output, target)


def plot_statistics(results_dict):
    """Use the dictionary with results for each trial and plot the mean, std and variance over trials

    Args:
        results_dict (int, list): Integer representing the trial along the list with the memory values
    """
    # Sum for each trial the memory values of each step
    results_dict_sum = {k: sum(v) for k, v in results_dict.items()}
    # Divide by the trials to get the mean of the memory values
    results_dict_mean = {k: v / args.trials for k, v in results_dict_sum.items()}
    
    # get the variance and std between each trial
    results_dict_var = np.var(list(results_dict_mean.values()))
    results_dict_std = np.std(list(results_dict_mean.values()))
    
    # plot the mean, std and variance
    plt.figure(figsize=(12, 6))
    plt.plot(list(results_dict_mean.keys()), list(results_dict_mean.values()), label="Mean")
    plt.fill_between(list(results_dict_mean.keys()), list(results_dict_mean.values()) - results_dict_std, list(results_dict_mean.values()) + results_dict_std, alpha=0.3, label="Std")
    plt.fill_between(list(results_dict_mean.keys()), list(results_dict_mean.values()) - results_dict_var, list(results_dict_mean.values()) + results_dict_var, alpha=0.3, label="Var")
    plt.grid(True, which="both", linestyle="--")
    plt.xlabel("Delay")
    plt.ylabel("Memory Capacity")
    plt.title("Memory Capacity over delay steps")
    plt.legend()


    return plt
 
@torch.no_grad()
def test(dataset, target, classifier, scaler):
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    activations = scaler.transform(activations)
    prediction = classifier.predict(activations)
    
    # wandb logs
    if args.wandb:
        wandb.log({"prediction": prediction, "target": target})
    
    error = criterion_eval(torch.tensor(prediction), torch.tensor(target))
    #nrmse_error = nrmse(prediction, target)
    
    return error
 
    
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_memory, valid_memory, test_memory = 0.0, 0.0, 0.0
train_nrmse, valid_nrmse, test_nrmse = 0.0, 0.0, 0.0

train_memory_dict, valid_memory_dict, test_memory_dict = defaultdict(list), defaultdict(list), defaultdict(list)
train_nrmse_list, valid_nrmse_list, test_nrmse_list = [], [], []

for t in range(args.trials):
    if args.esn:
        model = DeepReservoir(
            n_inp,
            tot_units=args.n_hid,
            n_layers=args.n_layers,
            spectral_radius=args.rho,
            input_scaling=args.inp_scaling,
            connectivity_recurrent=int((1 - args.sparsity) * args.n_hid),
            connectivity_input=args.n_hid,
            leaky=args.leaky,
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma, 
            epsilon,
            args.diffusive_gamma,
            args.rho,
            args.inp_scaling,
            args.topology,
            args.n_hid_layers,
            device=device,
        ).to(device)
    elif args.deepron:
        model = DeepRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.n_hid_layers,
            args.dt,
            gamma,
            epsilon,
            args.diffusive_gamma,
            args.rho,
            args.inp_scaling,
            device=device,
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")

    for i in range(delay):
        (
            (train_dataset, train_target),
            (test_dataset, test_target) 
            # since we iterate from 0 we need to add 1 to the i in the cycle
        ) = get_memory_capacity(delay=i+1, test_size=1000)

        # apply washout to the targets
        train_target = train_target[washout:]
        #valid_target = valid_target[washout:]
        test_target = test_target[washout:]
        
        dataset = train_dataset.reshape(1, -1, 1).to(device)
        target = train_target.reshape(-1, 1).numpy()
        activations = model(dataset)[0].cpu().numpy()
        activations = activations[:, washout:]
        activations = activations.reshape(-1, args.n_hid)
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
        classifier = Ridge(max_iter=1000).fit(activations, target)
        
        train_memory = test(train_dataset, train_target, classifier, scaler)
        test_memory = (
            test(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
        )
        
        # check shapes or type coming out of model
        print("Train memory: ", train_memory, "Test memory: ", test_memory)
        
        train_memory_dict[i].append(train_memory)
        test_memory_dict[i].append(test_memory)
        
        print(
            f"Trial {t}, delay {i+1}/{delay}, "  
            f"train memory: {round(train_memory, 2)} "
            f"test memory: {round(test_memory, 2)}"
        )
    
if args.ron:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.deepron:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_DEEPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"MemoryCapacity_log_ESN{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

# sum train, valid and test memory dict lists and divide by the number of trials
train_memory = sum([sum(v) for k, v in train_memory_dict.items()]) / args.trials
test_memory = sum([sum(v) for k, v in test_memory_dict.items()]) / args.trials

plt = plot_statistics(train_memory_dict)
plt_test = plot_statistics(test_memory_dict)
plotly_fig = tls.mpl_to_plotly(plt.gcf())
plotly_fig_test = tls.mpl_to_plotly(plt_test.gcf())

if args.wandb:
    # save the plot as a wandb artifact
    wandb.log({"Memory Capacity": plotly_fig})
    wandb.log({"Memory Capacity Test": plotly_fig_test})
    plt.savefig(os.path.join(args.resultroot, f"MemoryCapacity_plot{args.resultsuffix}{args.delay}.png"))
    plt_test.savefig(os.path.join(args.resultroot, f"MemoryCapacity_test_plot{args.resultsuffix}{args.delay}.png"))
    wandb.log({"train_memory": train_memory, "test_memory": test_memory})
    # TODO plotly broken
    #for i in range(args.trials):
        #wandb.Image(train_memory_list[::delay], caption="Debug MC plot")
    
ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, " 
ar += (
    f"Memory capacity for train: {train_memory} for test: {test_memory}"
)
f.write(ar + "\n")
f.close()
