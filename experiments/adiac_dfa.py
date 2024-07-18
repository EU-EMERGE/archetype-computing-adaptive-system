import torch
import warnings
import os
import numpy as np
from tqdm import tqdm
import argparse
from acds.benchmarks import get_adiac_data
from acds.archetypes.utils import count_parameters
from typing import List
from acds.archetypes import RNN_DFA, GRU_DFA

parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument('--modelname', type=str, default="rnn", choices=["rnn", "gru"],
                    help="Model name to use")
parser.add_argument("--batch", type=int, default=1000, help="batch size")

parser.add_argument("--cpu", action="store_true")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--grad_clip', type=float, default=0, help="Min-Max value clipping")

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
def test(model, data_loader):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        y = y.to(device).long()
        output = model(x)
        activations.append(output.cpu())
        ys.append(y.cpu())
    activations = torch.cat(activations, dim=0)
    ys = torch.cat(ys, dim=0)
    return (torch.argmax(activations, dim=-1) == ys).float().mean().item() * 100


n_inp = 1
n_out = 37

if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."
    train_loader, valid_loader, test_loader = get_adiac_data(
        args.dataroot, args.batch, args.batch, whole_train=True
    )
else:
    train_loader, valid_loader, test_loader = get_adiac_data(
        args.dataroot, args.batch, args.batch
    )

max_test_accs: List[float] = []
if args.trials > 1:
    assert args.use_test, "Multiple runs are only for the final test phase with the test set."

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):
    if args.modelname == 'rnn':
        model = RNN_DFA(n_inp, args.n_hid, n_out, args.grad_clip, device).to(device)
    elif args.modelname == 'gru':
        model = GRU_DFA(n_inp, args.n_hid, n_out, args.grad_clip, device).to(device)
    else:
        raise ValueError("Wrong model name")

    criterion = torch.nn.NLLLoss()  # only used for reporting
    total_params, total_trainable_params = count_parameters(model)
    print(f"Total parameters model: {total_params}")
    print(f"Total trainable parameters model: {total_trainable_params}")

    max_valid_acc = 0.
    for epoch in range(args.epochs):
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device).long()
            output, hidden, dW, dV, dbW, error = model(x, y=y.squeeze(-1))
            model.update_weights(hidden, dW, dV, dbW, error, args.lr)
            loss = criterion(torch.log(output), y.squeeze(-1))
        train_acc = test(model, train_loader)
        acc = test(model, valid_loader) if not args.use_test else test(model, test_loader)
        max_valid_acc = max(max_valid_acc, acc)
        print(f"Epoch {epoch}, train accuracy {train_acc}, valid/test accuracy: {acc}")

    train_acc = test(model, train_loader)
    if args.use_test:
        test_acc = max(max_valid_acc, test(model, test_loader))
        valid_acc = 0.0
    else:
        valid_acc = max(max_valid_acc, test(model, valid_loader))
        test_acc = 0.0

    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

f = open(os.path.join(args.resultroot, f"AdiacDFA_{args.modelname}_log_{args.resultsuffix}.txt"), "a")

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
