import argparse
import warnings
from typing import List
import os
import numpy as np
import torch.nn.utils
from tqdm import tqdm
from acds.archetypes.utils import count_parameters

from acds.archetypes import (
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    TrainedPhysicallyImplementableRandomizedOscillatorsNetwork,
    hcoRNN
)
from acds.benchmarks import get_mackey_glass_windows

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument('--modelname', type=str, default="pron", choices=["pron", "trainedpron", "hcornn"],
                    help="Model name to use")
parser.add_argument("--train_oscillators", action="store_true")
parser.add_argument("--train_recurrent", action="store_true")
parser.add_argument("--batch", type=int, default=30, help="batch size")
parser.add_argument("--lag", type=int, default=1, help="prediction lag")
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


criterion_eval = torch.nn.L1Loss(reduction='mean')
criterion_train = torch.nn.MSELoss()


@torch.no_grad()
def test(data_loader, readout):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        y = y.to(device)
        output = model(x)[-1][0]
        activations.append(output)
        ys.append(y)
    activations = torch.cat(activations, dim=0)
    ys = torch.cat(ys, dim=0)
    out = readout(activations)
    error = criterion_eval(out, ys).item()
    return error


n_inp = 1
n_out = 1
gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

min_test_loss: List[float] = []
if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."
train_loader, valid_loader, test_loader = get_mackey_glass_windows(args.dataroot, chunk_length=50,
                                                                   prediction_lag=args.lag,
                                                                   tr_bs=args.batch)

train_losses, valid_losses, test_losses = [], [], []
for i in range(args.trials):
    if args.modelname == 'pron':
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
    elif args.modelname == 'trainedpron':
        model = TrainedPhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            device=device,
            matrix_friction=args.matrix_friction,
            train_oscillators=args.train_oscillators,
            train_recurrent=args.train_recurrent
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
    criterion = torch.nn.MSELoss()

    total_params, total_trainable_params = count_parameters(model)
    total_params_readout, total_trainable_params_readout = count_parameters(readout)
    print(f"Total parameters model/readout: {total_params}/{total_params_readout}")
    print(f"Total trainable parameters model/readout: {total_trainable_params}/{total_trainable_params_readout}")

    min_valid_loss = 100000.
    for epoch in range(args.epochs):
        for x, y in tqdm(train_loader):
            optimizer_res.zero_grad()
            optimizer_readout.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model(x)[-1][0]
            output = readout(output)
            loss = criterion_train(output, y)
            loss.backward()
            optimizer_res.step()
            optimizer_readout.step()
        train_loss = test(train_loader, readout)
        val_loss = test(valid_loader, readout) if not args.use_test else test(test_loader, readout)
        min_valid_loss = min(min_valid_loss, val_loss)
        print(f"Epoch {epoch}, train loss {train_loss}, valid/test loss: {val_loss}")

    train_loss = test(train_loader, readout)
    if args.use_test:
        test_loss = min(min_valid_loss, test(test_loader, readout))
        valid_loss = 10000.
    else:
        valid_loss = min(min_valid_loss, test(valid_loader, readout))
        test_loss = 10000.

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    test_losses.append(test_loss)

f = open(os.path.join(args.resultroot, f"TrainedMG_log_{args.modelname}{args.resultsuffix}.txt"), "a")

ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_loss, 2)) for train_loss in train_losses]} "
    f"valid: {[str(round(valid_loss, 2)) for valid_loss in valid_losses]} "
    f"test: {[str(round(test_loss, 2)) for test_loss in test_losses]}"
    f"mean/std train: {np.mean(train_losses), np.std(train_losses)} "
    f"mean/std valid: {np.mean(valid_losses), np.std(valid_losses)} "
    f"mean/std test: {np.mean(test_losses), np.std(test_losses)}"
)
f.write(ar + "\n")
f.close()
