import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork
)
# Import memory capacity
from acds.benchmarks import get_memory_capacity

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--resultroot", type=str)

parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")

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
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)

parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")
parser.add_argument("--topology", type=str, default="full", choices=["full", "antisymmetric"], help="Topology of the hidden-to-hidden matrix")

parser.add_argument("--use_test", action="store_true")
parser.add_argument("--trials", type=int, default=1)

parser.add_argument("--resultsuffix", type=str, default="")

args = parser.parse_args()

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

@torch.no_grad()
def test(data_loader, classifier, scaler):
    # Test classifier using memory capacity test
    # Memory capacity loop over k steps of lag
    # sums the squared correlation coefficient between the target signal and the predicted signal
    # returns the sum of the squared correlation coefficient
    # TODO: Implement the test function
    pass 

def square_correlation(output, target):
    return np.corrcoef(output.flatten(), target.flatten())[0, 1]**2

def nrmse(output, target):
    mse = np.mean((output - target)**2)
    rms_target = np.sqrt(np.mean(target**2))
    return np.sqrt(mse) / rms_target

n_inp = 1
n_out = 1
washout = 100

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_nrmse, valid_nrmse, test_nrmse = [], [], []
memory_capacity,  = []
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
            device=device,
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")

    (
        (train_dataset, train_target),
        (valid_dataset, valid_target), 
        (test_dataset, test_target) 
    ) = get_memory_capacity(i, washout=washout, train_ratio=0.8, test_size=1000)


    dataset = train_dataset.reshape(1, -1, 1).to(device)
    target = train_target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = Ridge(max_iter=1000).fit(activations, target)
    train_nmse = test(train_dataset, train_target, classifier, scaler)
    valid_nmse = (
        test(valid_dataset, valid_target, classifier, scaler)
        if not args.use_test
        else 0.0
    )
    test_nmse = (
        test(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
    )
    train_mse.append(train_nmse)
    valid_mse.append(valid_nmse)
    test_mse.append(test_nmse)

if args.ron:
    f = open(os.path.join(args.resultroot, f"MG_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.deepron:
    f = open(os.path.join(args.resultroot, f"MG_log_DEEPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"MG_log_ESN{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_mse]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_mse]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_mse]}"
    f"mean/std train: {np.mean(train_mse), np.std(train_mse)} "
    f"mean/std valid: {np.mean(valid_mse), np.std(valid_mse)} "
    f"mean/std test: {np.mean(test_mse), np.std(test_mse)}"
)
f.write(ar + "\n")
f.close()
