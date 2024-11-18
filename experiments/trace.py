import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork
)
from acds.benchmarks import get_trace_data


parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--resultroot", type=str)

parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")

parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--n_hid", type=int, default=50)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--gamma_range", type=float, default=0.5)
parser.add_argument("--epsilon_range", type=float, default=1)
parser.add_argument("--rho", type=float)
parser.add_argument("--inp_scaling", type=float)
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
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

n_inp = 1
n_out = 4

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

if args.trials > 1:
    assert args.use_test, "Multiple trials are only for the test phase."
    train_loader, valid_loader, test_loader = get_trace_data(args.batch, 30, whole_train = True)
else:
    train_loader, valid_loader, test_loader = get_trace_data(args.batch, 30)

train_accs, valid_accs, test_accs = [], [], []

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

    activations, ys = [], []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()

    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=10000).fit(activations, ys)

    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) if args.use_test else 0.0
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

if args.ron:
    f = open(os.path.join(args.resultroot, f"Trace_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"Trace_log_ESN{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()