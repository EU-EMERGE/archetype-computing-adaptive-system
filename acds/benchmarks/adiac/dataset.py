from typing import Tuple

import torch
from torch.nn import functional as F


class AdiacDataset(torch.utils.data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """Initialize the dataset.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): Tuple of torch tensors, where the first
                element is the input and the second element is the target.
        """
        self.data = data

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=37).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self) -> int:
        return len(self.data)
