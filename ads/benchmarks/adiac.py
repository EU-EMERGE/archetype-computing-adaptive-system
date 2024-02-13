import torch
import numpy as np
import os
from torch.utils import data
from torch.nn import functional as F


class datasetforRC(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


class Adiac_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=37).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)

def get_Adiac_data(root_path, bs_train,bs_test, whole_train=False, RC=True):
    train_name = 'Adiac_TRAIN.txt'
    test_name = 'Adiac_TEST.txt'

    def fromtxt_to_numpy(filename=train_name, valid_len=120):
        # read the txt file
        adiacdata = np.genfromtxt(os.path.join(root_path, filename), dtype='float64')
        # create a list of lists with each line of the txt file
        l = []
        for i in adiacdata:
            el = list(i)
            while len(el) < 3:
                el.append('a')
            l.append(el)
        # create a numpy array from the list of lists
        arr = np.array(l)
        if valid_len is None:
            test_targets = arr[:,0]-1
            test_series = arr[:,1:]
            return test_series, test_targets
        else:
            if valid_len == 0:
                train_targets = arr[:,0]-1
                train_series = arr[:,1:]
                val_targets = arr[0:0,0] # empty
                val_series = arr[0:0,1:] # empty
            elif valid_len > 0 :
                train_targets = arr[:-valid_len,0]-1
                train_series = arr[:-valid_len,1:]
                val_targets = arr[-valid_len:,0]-1
                val_series = arr[-valid_len:,1:]
            return train_series, train_targets, val_series, val_targets

    # Generate list of input-output pairs
    def inp_out_pairs(data_x, data_y):
        mydata = []
        for i in range(len(data_y)):
            sample = (data_x[i,:], data_y[i])
            mydata.append(sample)
        return mydata

    # generate torch datasets
    if whole_train:
        valid_len = 0
    else:
        valid_len = 120
    train_series, train_targets, val_series, val_targets = fromtxt_to_numpy(filename=train_name, valid_len=valid_len)
    mytraindata, myvaldata = inp_out_pairs(train_series, train_targets), inp_out_pairs(val_series, val_targets)
    if RC:
        mytraindata, myvaldata = datasetforRC(mytraindata), datasetforRC(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename=test_name, valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = datasetforRC(mytestdata)
    else:
        mytraindata, myvaldata = Adiac_dataset(mytraindata), Adiac_dataset(myvaldata)
        test_series, test_targets = fromtxt_to_numpy(filename=test_name, valid_len=None)
        mytestdata = inp_out_pairs(test_series, test_targets)
        mytestdata = Adiac_dataset(mytestdata)


    # generate torch dataloaders
    mytrainloader = data.DataLoader(mytraindata,
                    batch_size=bs_train, shuffle=True, drop_last=False)
    myvaloader = data.DataLoader(myvaldata,
                        batch_size=bs_test, shuffle=False, drop_last=False)
    mytestloader = data.DataLoader(mytestdata,
                batch_size=bs_test, shuffle=False, drop_last=False)
    return mytrainloader, myvaloader, mytestloader