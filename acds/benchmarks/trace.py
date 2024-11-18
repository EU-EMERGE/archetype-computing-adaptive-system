import numpy as np

import torch
from torch.utils.data import DataLoader
from aeon.datasets import load_classification

def get_trace_data(
    bs_train: int, 
    bs_test: int, 
    whole_train: bool = False, 
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the Trace dataset from time series classification website and return the train, validation and test
    dataloaders.

    Args:
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        whole_train (bool, optional): If True, the whole dataset is used for training.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders.
    """

    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (torch.tensor(data_x[i, :], dtype=torch.float32), torch.tensor(int(data_y[i]), dtype=torch.long))
            mydata.append(sample)
        return mydata
    

    x, y = load_classification("Trace")

    arr_data = np.array(x)
    arr_data = arr_data.transpose(0, 2, 1)
    arr_targets = np.array(y)  

    if whole_train:
        valid_len = 0
    else:        
        valid_len = 30  # 30% for validation. 

    train_idx = 100 - valid_len

    train_series = arr_data[:train_idx]
    train_targets = arr_targets[:train_idx].astype(float) - 1.0

    valid_series = arr_data[train_idx:100]
    valid_targets = arr_targets[train_idx:100].astype(float) - 1.0

    test_series = arr_data[100:]
    test_targets = arr_targets[100:].astype(float) - 1.0

    train_data, eval_data, test_data = inp_out_pairs(train_series, train_targets), inp_out_pairs(valid_series, valid_targets), inp_out_pairs(test_series, test_targets)
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
