import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from acds.benchmarks.rc_dataset import RCDataset

from .dataset import AdiacDataset


def get_adiac_data(
    root_path: os.PathLike,
    bs_train: int,
    bs_test: int,
    whole_train: bool = False,
    for_rc: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the ADIAC dataset from a txt file and return the train, validation and test
    dataloaders.

    Args:
        root_path (os.PathLike): Path to the folder containing the txt files.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        whole_train (bool, optional): If True, the whole dataset is used for training.
            Defaults to False.
        for_rc (bool, optional): If True, the data is returned as a RCDataset. Defaults
            to True.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders.
    """

    def fromtxt_to_numpy(train: bool, valid_len: Optional[int] = 120):
        # read the txt file
        adiac_path = Path(root_path) / f"{'train' if train else 'test'}.txt"
        adiacdata = np.genfromtxt(adiac_path, dtype="float64")
        # create a list of lists with each line of the txt file
        l = []
        for i in adiacdata:
            el = list(i)
            while len(el) < 3:
                el.append("a")
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = arr[:, 0] - 1
            test_series = arr[:, 1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = arr[:, 0] - 1
                train_series = arr[:, 1:]
                val_targets = arr[0:0, 0]  # empty
                val_series = arr[0:0, 1:]  # empty
            elif valid_len > 0:
                train_targets = arr[:-valid_len, 0] - 1
                train_series = arr[:-valid_len, 1:]
                val_targets = arr[-valid_len:, 0] - 1
                val_series = arr[-valid_len:, 1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i, :], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    if whole_train:
        valid_len = 0
    else:
        valid_len = 120
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(
        train=True, valid_len=valid_len
    )
    train_data, eval_data = inp_out_pairs(train_series, train_targets), inp_out_pairs(
        val_series, val_targets
    )

    data_builder = RCDataset if for_rc else AdiacDataset
    train_data, eval_data = data_builder(train_data), data_builder(eval_data)
    test_series, test_targets = fromtxt_to_numpy(train=False, valid_len=None)
    test_data = inp_out_pairs(test_series, test_targets)
    test_data = data_builder(test_data)

    # generate torch dataloaders
    train_loader = DataLoader(
        train_data, batch_size=bs_train, shuffle=True, drop_last=False
    )
    eval_loader = DataLoader(
        eval_data, batch_size=bs_test, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_data, batch_size=bs_test, shuffle=False, drop_last=False
    )
    return train_loader, eval_loader, test_loader
