import numpy as np
from sklearn.model_selection import train_test_split
import torch
from acds.archetypes import RandomizedOscillatorsNetwork, DeepReservoir
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


def normalize_targets(d, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)
    d_normalized = (d - mean) / std
    return d_normalized, mean, std


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
    parser.add_argument('--state_average', action='store_true', help='Whether to average the hidden states over time')
    parser.add_argument('--arm', type=int, default=-1, choices=[-1, 0, 1, 2], help='Use only one input arm. -1 to use all.')
    args = parser.parse_args()

    dataset = np.load('data/dataset_squid.npz')

    fullx, fully = dataset['x'], dataset['y']
    # predict only center position (x, y)
    fully = fully[:, 1:3]

    # remove time info
    fullx = fullx[:, :, 1:]

    if args.arm >= 0:
        # select only feature number 5 7 and 9 from x
        fullx = fullx[:, :, [3+args.arm, 6+args.arm, 12+args.arm, 15+args.arm, 18+args.arm]]
    else:
        # use all arms
        fullx = fullx[:, :, [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

    # split in train, validation test with 60%, 15%, 15% ratio (165, 36, 36)
    x_train, x_temp, y_train, y_temp = train_test_split(fullx, fully, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    normalized_x_train, mean, std = normalize_features(x_train)
    normalized_x_val, _, _ = normalize_features(x_val, mean, std)
    normalized_x_test, _, _ = normalize_features(x_test, mean, std)

    normalized_y_train, meany, stdy = normalize_targets(y_train)
    normalized_y_val, _, _ = normalize_targets(y_val, meany, stdy)
    normalized_y_test, _, _ = normalize_targets(y_test, meany, stdy)

    # convert into Pytorch tensors
    train_data = torch.tensor(normalized_x_train, dtype=torch.float32)
    val_data = torch.tensor(normalized_x_val, dtype=torch.float32)
    test_data = torch.tensor(normalized_x_test, dtype=torch.float32)
    train_labels = torch.tensor(normalized_y_train, dtype=torch.float32)
    val_labels = torch.tensor(normalized_y_val, dtype=torch.float32)
    test_labels = torch.tensor(normalized_y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_inp = train_data.shape[-1]
    n_out = train_labels.shape[-1]
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

    with open('results.csv', 'a') as f:        
        f.write(f"{args.model},{args.arm},None,{args.n_hid},{args.rho},{args.gamma},{args.epsilon},{train_rmse:.4f},{valid_rmse:.4f},{test_rmse:.4f}\n")


    import matplotlib.pyplot as plt

    dataset_lens = [80000-300, 65000-300, 100000-300, 100000-300, 60000-300, 35000-300, 100000-300, 80000-300, 100000-300, 100000-300]
    normalized_x, _, _ = normalize_features(fullx, mean, std)
    normalized_y, _, _ = normalize_targets(fully, meany, stdy)
    data_dataset = torch.utils.data.TensorDataset(torch.tensor(normalized_x, dtype=torch.float32), 
                                                torch.tensor(normalized_y, dtype=torch.float32))
    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y.float())
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    preds = regressor.predict(activations)

    plot_one_point_every = 100
    start = 0
    for i, l in enumerate(dataset_lens):
        end = start + l
        xpred, ypred = preds[start:end, 0][::plot_one_point_every], preds[start:end, 1][::plot_one_point_every]
        x, y = ys[start:end, 0][::plot_one_point_every], ys[start:end, 1][::plot_one_point_every]
        
        base_color_pred = np.array([0.1, 0.2, 0.7])
        base_color_true = np.array([0.7, 0.2, 0.1])
        alpha = np.linspace(0.1, 1.0, len(x))

        colors_pred = np.ones((len(x), 4))
        colors_pred[:, :3] = base_color_pred
        colors_pred[:, 3] = alpha

        colors_true = np.ones((len(x), 4))
        colors_true[:, :3] = base_color_true
        colors_true[:, 3] = alpha

        plt.scatter(x, y, color=colors_true, s=3)
        plt.scatter(xpred, ypred, color=colors_pred, s=3)
        plt.title('Pred vs true, dataset ' + str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['True', 'Predicted'])
        plt.savefig(f'squid_predicted_trajectory_{i}.png')
        plt.close()
        start = end