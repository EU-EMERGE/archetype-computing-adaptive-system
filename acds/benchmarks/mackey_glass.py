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
