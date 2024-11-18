import argparse
import warnings
from typing import List
import os
import numpy as np
import torch.nn.utils
from tqdm import tqdm
from acds.archetypes.utils import count_parameters

from acds.archetypes import (
    TrainedPhysicallyImplementableRandomizedOscillatorsNetwork,
    hcoRNN
)
from acds.benchmarks import get_mnist_data

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument('--modelname', type=str, default="hcornn", choices=["trainedpron", "hcornn"],
                    help="Model name to use")
parser.add_argument("--train_oscillators", action="store_true")
parser.add_argument("--train_recurrent", action="store_true")
parser.add_argument("--batch", type=int, default=256, help="batch size")
parser.add_argument(
    "--dt", type=float, default=0.01, help="step size <dt> of the coRNN"
)
parser.add_argument(
    "--gamma", type=float, default=3
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=5,
)
parser.add_argument(
    "--gamma_range",
    type=float,
    default=2,
)
parser.add_argument(
    "--epsilon_range",
    type=float,
    default=1,
)
parser.add_argument("--cpu", action="store_true")

parser.add_argument("--matrix_friction", action="store_true")
parser.add_argument("--input_fn", type=str, default="linear", choices=["linear", "mlp"],
                    help="input preprocessing modality")

parser.add_argument("--diffusive_gamma", type=float, default=0.0, help="diffusive term")

parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)

parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

parser.add_argument(
    "--topology",
    type=str,
    default="orthogonal",
    choices=["orthogonal", "antisymmetric"],
    help="Topology of the h2h matrix",
)

args = parser.parse_args()

assert args.dataroot is not None, "No dataroot provided."
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."


device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)


@torch.no_grad()
def test(data_loader, readout):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        y = y.to(device).long()
        x = x.view(x.shape[0], 784, -1)
        output = model(x)[-1][0]
        activations.append(output)
        ys.append(y)
    activations = torch.cat(activations, dim=0)
    ys = torch.cat(ys, dim=0).squeeze(-1)
    return (torch.argmax(readout(activations), dim=-1) == ys).float().mean().item()


n_inp = 1
n_out = 10  # classes
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

max_test_accs: List[float] = []
if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."
train_loader, valid_loader, test_loader = get_mnist_data(args.dataroot, args.batch, 512)

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):
    if args.modelname == 'trainedpron':
        model = TrainedPhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            args.diffusive_gamma,
            gamma,
            epsilon,
            device=device,
            matrix_friction=args.matrix_friction,
            train_oscillators=args.train_oscillators,
            train_recurrent=args.train_recurrent,
            topology=args.topology
        ).to(device)
    elif args.modelname == 'hcornn':
        model = hcoRNN(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            device=device,
            matrix_friction=args.matrix_friction,
            train_oscillators=args.train_oscillators
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")

    readout = torch.nn.Linear(args.n_hid, n_out).to(device)
    optimizer_res = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_readout = torch.optim.Adam(readout.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    total_params, total_trainable_params = count_parameters(model)
    total_params_readout, total_trainable_params_readout = count_parameters(readout)
    print(f"Total parameters model/readout: {total_params}/{total_params_readout}")
    print(f"Total trainable parameters model/readout: {total_trainable_params}/{total_trainable_params_readout}")

    max_valid_acc = 0.
    for epoch in range(args.epochs):
        for x, y in tqdm(train_loader):
            optimizer_res.zero_grad()
            optimizer_readout.zero_grad()
            x = x.to(device)
            y = y.to(device).long()
            x = x.view(x.shape[0], 784, -1)
            output = model(x)[-1][0]
            output = readout(output)
            loss = criterion(output, y.squeeze(-1))
            loss.backward()
            optimizer_res.step()
            optimizer_readout.step()
        train_acc = test(train_loader, readout)
        acc = test(valid_loader, readout) if not args.use_test else test(test_loader, readout)
        max_valid_acc = max(max_valid_acc, acc)
        print(f"Epoch {epoch}, train accuracy {train_acc}, valid/test accuracy: {acc}")

    train_acc = test(train_loader, readout)
    if args.use_test:
        test_acc = max(max_valid_acc, test(test_loader, readout))
        valid_acc = 0.0
    else:
        valid_acc = max(max_valid_acc, test(valid_loader, readout))
        test_acc = 0.0

    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

if args.modelname == "trainedpron":
    f = open(os.path.join(args.resultroot, f"TrainedSMNIST_log_{args.modelname}{args.topology}{args.resultsuffix}.txt"), "a")
else:
    f = open(os.path.join(args.resultroot, f"TrainedSMNIST_log_{args.modelname}{args.resultsuffix}.txt"), "a")

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
