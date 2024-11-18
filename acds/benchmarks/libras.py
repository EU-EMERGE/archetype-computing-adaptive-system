import numpy as np

import torch
from torch.utils.data import DataLoader
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split

def get_libras_data(
    bs_train: int, 
    bs_test: int, 
    whole_train: bool = False, 
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the Libras dataset from time series classification website and return the train, validation and test
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
    

    x, y = load_classification("Libras")

    arr_data = np.array(x)
    arr_data = arr_data.transpose(0, 2, 1)
    arr_targets = np.array(y)  

    train_series = arr_data[:180]
    train_targets = arr_targets[:180].astype(float) - 1.0

    test_series = arr_data[180:]
    test_targets = arr_targets[180:].astype(float) - 1.0

    valid_series = []
    valid_targets = []
    if whole_train:
        '''indices = np.arange(len(test_series))
        np.random.seed(20) 
        np.random.shuffle(indices)
        test_series = test_series[indices]
        test_targets = test_targets[indices].astype(float) - 1.0'''
    else:        
        train_series, valid_series, train_targets, valid_targets = train_test_split(train_series, train_targets, test_size=0.3, random_state=20, stratify=train_targets)

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
