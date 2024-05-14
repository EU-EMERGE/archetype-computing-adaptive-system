import argparse
import warnings
from typing import List
import os
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_adiac_data

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument("--batch", type=int, default=30, help="batch size")
parser.add_argument(
    "--dt", type=float, default=0.042, help="step size <dt> of the coRNN"
)
parser.add_argument(
    "--gamma", type=float, default=2.7, help="y control parameter <gamma> of the coRNN"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument(
    "--gamma_range",
    type=float,
    default=2.7,
    help="y controle parameter <gamma> of the coRNN",
)
parser.add_argument(
    "--epsilon_range",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument("--cpu", action="store_true")
parser.add_argument('--pron', action="store_true")

parser.add_argument("--matrix_friction", action="store_true")
parser.add_argument("--input_fn", type=str, default="linear", choices=["linear", "mlp"],
                    help="input preprocessing modality")

parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)

parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

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
        output = model(x)[-1][0]
        activations.append(output)
        ys.append(y)
    activations = torch.cat(activations, dim=0)
    ys = torch.cat(ys, dim=0).squeeze(-1)
    return (torch.argmax(readout(activations), dim=-1) == ys).float().mean().item()


n_inp = 1
n_out = 37  # classes
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

max_test_accs: List[float] = []
if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."
    train_loader, valid_loader, test_loader = get_adiac_data(
        args.dataroot, args.batch, args.batch, whole_train=True
    )
else:
    train_loader, valid_loader, test_loader = get_adiac_data(
        args.dataroot, args.batch, args.batch
    )

train_accs, valid_accs, test_accs = [], [], []

for i in range(args.trials):
    if args.pron:
        model = PhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device,
            input_function=args.input_fn,
            matrix_friction=args.matrix_friction,
        ).to(device)
        readout = torch.nn.Linear(args.n_hid, n_out).to(device)
    else:
        raise ValueError("Wrong model choice.")

    optimizer_res = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_readout = torch.optim.Adam(readout.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for x, y in tqdm(train_loader):
            optimizer_res.zero_grad()
            optimizer_readout.zero_grad()
            x = x.to(device)
            y = y.to(device).long()
            output = model(x)[-1][0]
            output = readout(output)
            loss = criterion(output, y.squeeze(-1))
            loss.backward()
            optimizer_res.step()
            optimizer_readout.step()
        acc = test(valid_loader, readout) if not args.use_test else test(test_loader, readout)
        print(f"Epoch {epoch}, accuracy: {acc}")

    train_acc = test(train_loader, readout)
    valid_acc = test(valid_loader, readout) if not args.use_test else 0.0
    test_acc = test(test_loader, readout) if args.use_test else 0.0
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

if args.pron:
    f = open(os.path.join(args.resultroot, f"Adiac_log_PRON_trained{args.resultsuffix}.txt"), "a")
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
