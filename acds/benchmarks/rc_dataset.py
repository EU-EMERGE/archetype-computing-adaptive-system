from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
)

import torch

if TYPE_CHECKING:
    import numpy as np


class RCDataset(torch.utils.data.Dataset):
    """A torch dataset to ease training and inference with Reservoir Computing models.

    This class assumes data to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, data: List[Tuple[np.ndarray, int]]):
        """Initialize the dataset.

        Args:
            data (List[Tuple[np.ndarray, int]]): List of tuples, where the first
                element is the input and the second element is the target.
        """
        self.data = data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the input and the target.
        """
        sample = self.data[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.data)
