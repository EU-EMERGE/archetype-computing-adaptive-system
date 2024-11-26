from typing import Literal

import numpy as np
import torch
from torch import nn


def count_parameters(model):
    """Return total number of parameters and
    trainable parameters of a PyTorch model.
    """
    params = []
    trainable_params = []
    for p in model.parameters():
        params.append(p.numel())
        if p.requires_grad:
            trainable_params.append(p.numel())
    pytorch_total_params = sum(params)
    pytorch_total_trainableparams = sum(trainable_params)
    return pytorch_total_params, pytorch_total_trainableparams


def sparse_eye_init(M: int) -> torch.FloatTensor:
    """Generates an M x M matrix to be used as sparse identity matrix for the re-scaling
    of the sparse recurrent kernel in presence of non-zero leakage. The neurons are
    connected according to a ring topology, where each neuron receives input only from
    one neuron and propagates its activation only to one other neuron. All the non-zero
    elements are set to 1.

    Args:
        M (int): number of hidden units.

    Returns:
        torch.FloatTensor: MxM identity matrix.
    """
    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """Generates an M x N matrix to be used as sparse (input) kernel For each row only C
    elements are non-zero (i.e., each input dimension is projected only to C neurons).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1]

    Args:
        M (int): number of hidden units
        N (int): number of input units
        C (int): number of nonzero elements

    Returns:
        torch.FloatTensor: MxN dense matrix
    """
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype("f") - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1) -> torch.FloatTensor:
    """Generates an M x M matrix to be used as sparse recurrent kernel. For each column
    only C elements are non-zero (i.e., each recurrent neuron take sinput from C other
    recurrent neurons). The non-zero elements are generated randomly from a uniform
    distribution in [-1,1].

    Args:
        M (int): number of hidden units
        C (int): number of nonzero elements

    Returns:
        torch.FloatTensor: MxM dense matrix
    """
    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype("f") - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def spectral_norm_scaling(
    W: torch.FloatTensor, rho_desired: float
) -> torch.FloatTensor:
    """Rescales W to have rho(W) = rho_desired .

    Args:
        W (torch.FloatTensor): input matrix to be rescaled
        rho_desired (float): desired spectral radius

    Returns:
        torch.FloatTensor: rescaled matrix
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)

def antisymmetric_matrix(
    W: torch.FloatTensor  
) -> torch.FloatTensor:
    """Transforms W to have an antisymmetric matrix
    
    Args:
        W (torch.FloatTensor): input matrix to be transformed

    Returns:
        torch.FloatTensor: transformed matrix
    """
    return (W - W.mT)

def get_hidden_topology(
    n_hid: int,
    topology: Literal["full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"],
    sparsity: float,
    scaler: float,
) -> torch.FloatTensor:
    """Generates the hidden-to-hidden weight matrix according to the specified topology
    and sparsity.

    Args:
        n_hid (int): number of hidden units.
        topology (str): topology of the hidden-to-hidden weight matrix. Options
            are 'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz', 'antisymmetric'.
        sparsity (float): sparsity of the hidden-to-hidden weight matrix.
        scaler (float): scaling factor for the hidden-to-hidden weight matrix.

    Returns:
        torch.Tensor: hidden-to-hidden weight matrix.
    """

    def get_sparsity(A):
        n_hid = A.shape[0]
        sparsity = 100 * (n_hid**2 - np.count_nonzero(A)) / n_hid**2
        return sparsity
    assert sparsity >= 0 and sparsity < 1, "Sparsity must be in [0,1)"

    if topology == "full":
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
    elif topology == "lower":
        h2h = torch.tril(2 * torch.rand(n_hid, n_hid) - 1)
        if sparsity > 0:
            n_zeroed_diagonals = int(sparsity * n_hid)
            for i in range(n_hid - 1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == "orthogonal":
        rand = torch.rand(n_hid, n_hid)
        orth = torch.linalg.qr(rand)[0]
        identity = torch.eye(n_hid)
        if sparsity > 0:
            n_zeroed_rows = int(sparsity * n_hid)
            idxs = torch.randperm(n_hid)[:n_zeroed_rows].tolist()
            identity[idxs, idxs] = 0.0
        h2h = torch.matmul(identity, orth)
        get_sparsity(h2h.numpy())
    elif topology == "band":
        h2h = 2 * torch.rand(n_hid, n_hid) - 1
        if sparsity > 0:
            n_zeroed_diagonals = int(np.sqrt(sparsity) * n_hid)
            for i in range(n_hid - 1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
                h2h.diagonal(i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == "ring":
        # scaler = 1
        h2h = torch.zeros(n_hid, n_hid)
        for i in range(1, n_hid):
            h2h[i, i - 1] = 1
        h2h[0, n_hid - 1] = 1
        h2h = scaler * h2h
        get_sparsity(h2h.numpy())
    elif topology == "toeplitz":
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
    elif topology == "antisymmetric":
        h2h = torch.triu(torch.randn(n_hid, n_hid, dtype=torch.float32))
        h2h = antisymmetric_matrix(h2h)
    else:
        raise ValueError(
            "Invalid topology. Options are 'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz'"
        )
    return h2h
