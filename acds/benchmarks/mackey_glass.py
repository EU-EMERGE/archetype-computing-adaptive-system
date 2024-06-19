import os

import torch


def get_mackey_glass(csvfolder: os.PathLike, lag=84, washout=200):
    """Get the Mackey-Glass dataset and return the train, validation and test datasets
    as torch tensors.

    Args:
        csvfolder (os.PathLike): Path to the directory containing the mackey_glass.csv file.
        lag (int, optional): Number of time steps to look back. Defaults to 84.
        washout (int, optional): Number of time steps to discard. Defaults to 200.

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: Train, validation and test datasets.
    """
    with open(os.path.join(csvfolder, "mackey_glass.csv"), "r") as f:
        data_lines = f.readlines()[0]

    # 10k steps
    dataset = torch.tensor([float(el) for el in data_lines.split(",")]).float()

    end_train = int(dataset.shape[0] / 2)
    end_val = end_train + int(dataset.shape[0] / 4)
    end_test = dataset.shape[0]

    train_dataset = dataset[: end_train - lag]
    train_target = dataset[washout + lag : end_train]

    val_dataset = dataset[end_train : end_val - lag]
    val_target = dataset[end_train + washout + lag : end_val]

    test_dataset = dataset[end_val : end_test - lag]
    test_target = dataset[end_val + washout + lag : end_test]

    return (
        (train_dataset, train_target),
        (val_dataset, val_target),
        (test_dataset, test_target),
    )


def get_mackey_glass_windows(csvfolder: os.PathLike, chunk_length, prediction_lag=84, tr_bs=10):
    from sktime.split import SlidingWindowSplitter
    import numpy as np

    with open(os.path.join(csvfolder, "mackey_glass.csv"), "r") as f:
        data_lines = f.readlines()[0]

    # 10k steps
    sequence = np.array([float(el) for el in data_lines.split(",")])

    splitter = SlidingWindowSplitter(fh=prediction_lag, window_length=chunk_length, step_length=1)

    len_train = int(sequence.shape[0] / 2)
    len_val = int(sequence.shape[0] / 4)

    splits = splitter.split_series(sequence)
    windows, targets = [], []
    for x, y in splits:
        windows.append(x)
        targets.append(y)

    windows = torch.from_numpy(np.array(windows)).float()
    targets = torch.from_numpy(np.array(targets)).float().squeeze(-1)

    train_dataset, train_target = windows[:len_train], targets[:len_train]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_dataset, train_target), batch_size=tr_bs, shuffle=True, drop_last=False
    )
    val_dataset, val_target = windows[len_train:len_train+len_val], targets[len_train:len_train+len_val]
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_dataset, val_target), batch_size=500, shuffle=False, drop_last=False
    )
    test_dataset, test_target = windows[len_train+len_val:], targets[len_train+len_val:]
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_dataset, test_target), batch_size=500, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader
