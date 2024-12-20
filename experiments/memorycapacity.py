import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import warnings
import wandb

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
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)

parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric"], help="Topology of the hidden-to-hidden matrix")
parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)

parser.add_argument("--resultsuffix", type=str, default="")

args = parser.parse_args()

if args.wandb == True:
    wandb.init(project="deep-ron-thesis",
            config={"architecture": "DeepRON" if args.deepron else "RON" if args.ron else "ESN",
                    }
    )

# make sure that n_hid_layers is a list of integers
args.n_hid_layers = [int(x) for x in args.n_hid_layers.split(",")]
#print("type of n_hid_layers", type(args.n_hid_layers))
print("n_hid_layers after parsing:", args.n_hid_layers)
print("type of n_hid_layers after parsing:", type(args.n_hid_layers))

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
delay = args.delay

def square_correlation(output, target):
    return np.corrcoef(output.flatten(), target.flatten())[0, 1]**2

def nrmse(output, target):
    mse = np.mean((output - target)**2)
    rms_target = np.sqrt(np.mean(target**2))
    return np.sqrt(mse) / rms_target

# set custom criterion eval to square correlation
def criterion_eval(output, target):
    return square_correlation(output, target)

@torch.no_grad()
def test(dataset, target, classifier, scaler):
    # Test classifier using memory capacity test
    # Memory capacity loop over k steps of lag
    # sums the squared correlation coefficient between the target signal and the predicted signal
    # returns the sum of the squared correlation coefficient
    # TODO: Implement the test function
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    activations = scaler.transform(activations)
    prediction = classifier.predict(activations)
    
    # wandb logs
    wandb.log({"prediction": prediction, "target": target})

    
    error = criterion_eval(torch.tensor(prediction), torch.tensor(target))

    return error
 
    
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_memory, valid_memory, test_memory = 0.0, 0.0, 0.0
train_memory_list, valid_memory_list, test_memory_list = [], [], []
# for memory capacity test we run k times because we want iterate over k steps of lag
memory_test = False

if memory_test:
    # the number of trials is the same as the lag
    args.trials = delay

for i in range(args.trials):
    if args.esn:
        model = DeepReservoir(
            n_inp,
            tot_units=args.n_hid,
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
        (valid_dataset, valid_target), 
        (test_dataset, test_target) 
        # since we iterate from 0 we need to add 1 to the i in the cycle
    ) = get_memory_capacity(delay=i+1, train_ratio=0.8, test_size=1000)

    # apply washout to the targets
    train_target = train_target[washout:]
    valid_target = valid_target[washout:]
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
    valid_memory = (
        test(valid_dataset, valid_target, classifier, scaler)
        if not args.use_test
        else 0.0
    )
    test_memory = (
        test(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
    )
    
    train_memory += train_memory
    valid_memory += valid_memory
    test_memory += test_memory
    
    train_memory_list.append(train_memory)
    valid_memory_list.append(valid_memory)
    test_memory_list.append(test_memory)
     
    print(
        f"Trial {i+1}/{delay} "
        f"train memory: {round(train_memory, 2)} "
        f"valid memory: {round(valid_memory, 2)} "
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

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, " 
ar += (
    f"train: {[str(round(train_memory, 2)) for train_memory in train_memory_list]} "
    f"valid: {[str(round(valid_memory, 2)) for valid_memory in valid_memory_list]} "
    f"test: {[str(round(test_memory, 2)) for test_memory in test_memory_list]} "
    f"mean/std train: {np.mean(train_memory_list), np.std(train_memory_list)} "
    f"mean/std valid: {np.mean(valid_memory_list), np.std(valid_memory_list)} "
    f"mean/std test: {np.mean(test_memory_list), np.std(test_memory_list)}"
    f"Memory capacity for train: {train_memory}, valid: {valid_memory}, test: {test_memory}"
)
f.write(ar + "\n")
f.close()
