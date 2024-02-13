import torch
import numpy as np


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales W to have rho(W) = rho_desired

    :param W:
    :param rho_desired:
    :return:
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)


def get_hidden_topology(n_hid, topology, sparsity, scaler):
    def get_sparsity(A):
        n_hid = A.shape[0]
        sparsity = 100 * (n_hid ** 2 - np.count_nonzero(A)) / n_hid ** 2
        return sparsity

    if topology == 'full':
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
    elif topology == 'lower':
        h2h = torch.tril(2 * torch.rand(n_hid, n_hid) - 1)
        if sparsity > 0:
            n_zeroed_diagonals = int(sparsity * n_hid)
            for i in range(n_hid-1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == 'orthogonal':
        rand = torch.rand(n_hid, n_hid)
        orth = torch.linalg.qr(rand)[0]
        identity = torch.eye(n_hid)
        if sparsity > 0:
            n_zeroed_rows = int(sparsity * n_hid)
            idxs = torch.randperm(n_hid)[:n_zeroed_rows].tolist()
            identity[idxs, idxs] = 0.
        h2h = torch.matmul(identity, orth)
        get_sparsity(h2h.numpy())
    elif topology == 'band':
        h2h = 2*torch.rand(n_hid, n_hid)-1
        if sparsity > 0:
            n_zeroed_diagonals = int(np.sqrt(sparsity) * n_hid)
            for i in range(n_hid-1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
                h2h.diagonal(i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == 'ring':
        # scaler = 1
        h2h = torch.zeros(n_hid, n_hid)
        for i in range(1, n_hid):
            h2h[i, i - 1] = 1
        h2h[0, n_hid - 1] = 1
        h2h = scaler * h2h
        get_sparsity(h2h.numpy())
    elif topology == 'toeplitz':
        from scipy.linalg import toeplitz
        bandwidth = int(scaler)  # 5
        upperdiagcoefs = np.zeros(n_hid)
        upperdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs = np.zeros(n_hid)
        lowerdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs[0] = upperdiagcoefs[0]  # diagonal coefficient
        h2h = toeplitz(list(lowerdiagcoefs), list(upperdiagcoefs))
        get_sparsity(h2h)
        h2h = torch.Tensor(h2h)
    else:
        raise ValueError("Wrong topology choice.")
    return h2h