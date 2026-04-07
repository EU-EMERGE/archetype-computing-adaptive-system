import numpy as np
from sklearn.model_selection import train_test_split
import torch
from ron import RandomizedOscillatorsNetwork
from esn import DeepReservoir
import argparse
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error


def normalize_features(d, mean=None, std=None):
    n_samples, timesteps, n_features = d.shape
    d_reshaped = d.reshape(-1, n_features)  # reshape to (n_samples * timesteps, n_features)
    if mean is None and std is None:
        mean = np.mean(d_reshaped, axis=0)
        std = np.std(d_reshaped, axis=0)
    d_normalized = (d_reshaped - mean) / std
    return d_normalized.reshape(n_samples, timesteps, n_features), mean, std


def normalize_targets(d, mean, std):
    d_normalized = (d - mean) / std
    return d_normalized


@torch.no_grad()
def test(data_loader, regressor, scaler, state_average=False):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        if state_average:
            output = model(x)[0].mean(dim=1)
            # weights = torch.linspace(0.1, 1.0, output.shape[1])
            # output = (output * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        else:
            output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y.float())
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    preds = regressor.predict(activations)
    rmse = root_mean_squared_error(ys, preds)
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RON model on touch dataset')
    parser.add_argument('--n_hid', type=int, default=100, help='Number of hidden units')
    parser.add_argument('--rho', type=float, default=0.9, help='Spectral radius')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--model', type=str, default='ron', choices=['ron', 'esn'], help='Model type')
    parser.add_argument('--leaky', type=float, default=1.0, help='Leaky factor for ESN (ignored if model is RON)')
    parser.add_argument('--ridge_alpha', type=float, default=1.0, help='Ridge regression alpha')
    parser.add_argument('--gamma', type=float, default=0.1, help='Damping factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Stiffness factor')
    parser.add_argument('--gamma_range', type=float, default=0.01, help='Range for gamma')
    parser.add_argument('--epsilon_range', type=float, default=0.01, help='Range for epsilon')
    parser.add_argument('--input_scaling', type=float, default=1.0, help='Input scaling factor')
    parser.add_argument('--use_test', action='store_true', help='Use test set for evaluation')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for test/validation')
    parser.add_argument('--noxplus', action='store_true', help='Whether to exclude X+ from the dataset')
    parser.add_argument('--state_average', action='store_true', help='Whether to average the hidden states over time')
    parser.add_argument('--predict_feature', type=int, default=-1, help='Only predict that feature. -1 to predict all features')
    args = parser.parse_args()

    if args.noxplus:
        dataset = np.load('data/dataset_cobot_noX+.npz')
    else:
        dataset = np.load('data/dataset_cobot.npz')

    x, y = dataset['x'], dataset['y']

    if args.predict_feature >= 0:
        x, y = np.expand_dims(x[:, :, args.predict_feature], -1), np.expand_dims(y[:, args.predict_feature], -1)  # only use the specified feature of the input and output

    labels = dataset['l']  # original file of data, preprocessing clarifies the mapping
    # split in train, validation test with 60%, 15%, 15% ratio (165, 36, 36)
    x_train, x_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(x, y, labels, test_size=0.3, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test, labels_val, labels_test = train_test_split(x_temp, y_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp)

    normalized_x_train, mean, std = normalize_features(x_train)
    normalized_x_val, _, _ = normalize_features(x_val, mean, std)
    normalized_x_test, _, _ = normalize_features(x_test, mean, std)

    normalized_y_train = normalize_targets(y_train, mean, std)
    normalized_y_val = normalize_targets(y_val, mean, std)
    normalized_y_test = normalize_targets(y_test, mean, std)

    # convert into Pytorch tensors
    train_data = torch.tensor(normalized_x_train, dtype=torch.float32)
    train_labels = torch.tensor(normalized_y_train, dtype=torch.float32)
    val_data = torch.tensor(normalized_x_val, dtype=torch.float32)
    val_labels = torch.tensor(normalized_y_val, dtype=torch.float32)
    test_data = torch.tensor(normalized_x_test, dtype=torch.float32)
    test_labels = torch.tensor(normalized_y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_inp = train_data.shape[-1]
    n_out = n_inp
    gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
    epsilon = (
        args.epsilon - args.epsilon_range / 2.0,
        args.epsilon + args.epsilon_range / 2.0,
    )

    if args.model == 'ron':
        model = RandomizedOscillatorsNetwork(
            n_inp=n_inp,
            n_hid=args.n_hid,
            dt=args.dt,
            gamma=args.gamma,
            epsilon=args.epsilon,
            rho=args.rho,
            input_scaling=args.input_scaling,
            device=device)
    else:
        model = DeepReservoir(
            input_size=n_inp,
            tot_units=args.n_hid,
            spectral_radius=args.rho,
            input_scaling=args.input_scaling,
            leaky=args.leaky,
            connectivity_recurrent=args.n_hid,
            connectivity_input=args.n_hid
        )
    model.to(device)


    activations, ys = [], []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        if args.state_average:
            output = model(x)[0]
            output = output.mean(dim=1)
            # weights = torch.linspace(0.1, 1.0, 100).to(device)
            # output = (output * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        else:
            output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y.float())
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    regressor = Ridge(alpha=args.ridge_alpha).fit(activations, ys)
    train_rmse = test(train_loader, regressor, scaler, state_average=args.state_average)
    valid_rmse = test(valid_loader, regressor, scaler, state_average=args.state_average) if not args.use_test else 0.0
    test_rmse = test(test_loader, regressor, scaler, state_average=args.state_average) if args.use_test else 0.0

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {valid_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
