import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from acds.archetypes import RandomizedOscillatorsNetwork
import argparse
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def normalize_features(d, mean=None, std=None):
    n_samples, timesteps, n_features = d.shape
    d_reshaped = d.reshape(-1, n_features)  # reshape to (n_samples * timesteps, n_features)
    if mean is None and std is None:
        mean = np.mean(d_reshaped, axis=0)
        std = np.std(d_reshaped, axis=0)
    d_normalized = (d_reshaped - mean) / std
    return d_normalized.reshape(n_samples, timesteps, n_features), mean, std


@torch.no_grad()
def test(data_loader, classifier, scaler, label_encoder=None):
    activations, ys = [], []
    for x, y in tqdm(data_loader):
        x = x.to(device)
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    preds = classifier.predict(activations)
    if label_encoder is not None:
        ys_dec = label_encoder.inverse_transform(ys)
        preds_dec = label_encoder.inverse_transform(preds)
    else:
        ys_dec = ys
        preds_dec = preds
    acc = accuracy_score(ys_dec, preds_dec)
    return acc, ys_dec, preds_dec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RON classifier on cobot dataset')
    parser.add_argument('--n_hid', type=int, default=100, help='Number of hidden units')
    parser.add_argument('--rho', type=float, default=0.9, help='Spectral radius')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularization for LogisticRegression')
    parser.add_argument('--gamma', type=float, default=0.1, help='Damping factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Stiffness factor')
    parser.add_argument('--gamma_range', type=float, default=0.01, help='Range for gamma')
    parser.add_argument('--epsilon_range', type=float, default=0.01, help='Range for epsilon')
    parser.add_argument('--input_scaling', type=float, default=1.0, help='Input scaling factor')
    parser.add_argument('--use_test', action='store_true', help='Use test set for evaluation')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for test/validation')
    args = parser.parse_args()

    dataset = np.load('data/dataset_cobot.npz')
    
    # y not used for classification
    x, y = dataset['x'], dataset['y']
    labels = dataset['l']

    # split in train, validation test with 60%, 15%, 15% ratio
    x_train, x_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(
        x, y, labels, test_size=0.3, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test, labels_val, labels_test = train_test_split(
        x_temp, y_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp)

    normalized_x_train, mean, std = normalize_features(x_train)
    normalized_x_val, _, _ = normalize_features(x_val, mean, std)
    normalized_x_test, _, _ = normalize_features(x_test, mean, std)

    # encode labels in case they are strings
    le = LabelEncoder().fit(labels_train)
    labels_train_enc = le.transform(labels_train)
    labels_val_enc = le.transform(labels_val)
    labels_test_enc = le.transform(labels_test)

    # convert into Pytorch tensors
    train_data = torch.tensor(normalized_x_train, dtype=torch.float32)
    train_labels = torch.tensor(labels_train_enc, dtype=torch.long)
    val_data = torch.tensor(normalized_x_val, dtype=torch.float32)
    val_labels = torch.tensor(labels_val_enc, dtype=torch.long)
    test_data = torch.tensor(normalized_x_test, dtype=torch.float32)
    test_labels = torch.tensor(labels_test_enc, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_inp = train_data.shape[-1]
    gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
    epsilon = (
        args.epsilon - args.epsilon_range / 2.0,
        args.epsilon + args.epsilon_range / 2.0,
    )
    model = RandomizedOscillatorsNetwork(
        n_inp=n_inp,
        n_hid=args.n_hid,
        dt=args.dt,
        gamma=args.gamma,
        epsilon=args.epsilon,
        rho=args.rho,
        input_scaling=args.input_scaling,
        device=device)
    model.to(device)

    # collect activations for training
    activations, ys = [], []
    for x_batch, y_batch in tqdm(train_loader):
        x_batch = x_batch.to(device)
        output = model(x_batch)[-1][0]
        activations.append(output.cpu())
        ys.append(y_batch)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()

    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)

    classifier = LogisticRegression(C=args.C).fit(activations, ys)

    train_acc, _, _ = test(train_loader, classifier, scaler, label_encoder=None)
    valid_acc = test(valid_loader, classifier, scaler, label_encoder=None)[0] if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler, label_encoder=None)[0] if args.use_test else 0.0

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    acc, ys_true, ys_pred = test(test_loader if args.use_test else valid_loader, classifier, scaler, label_encoder=None)
    print("Classification report:")
    print(classification_report(ys_true, ys_pred, target_names=[str(c) for c in le.classes_]))
