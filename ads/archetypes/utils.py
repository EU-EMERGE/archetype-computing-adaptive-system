import torch
import numpy as np


def sparse_eye_init(M: int) -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse identity matrix for the
    re-scaling of the sparse recurrent kernel in presence of non-zero leakage.
    The neurons are connected according to a ring topology, where each neuron
    receives input only from one neuron and propagates its activation only to
    one other neuron. All the non-zero elements are set to 1.

    :param M: number of hidden units
    :return: dense weight matrix
    """
    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x N matrix to be used as sparse (input) kernel
    For each row only C elements are non-zero (i.e., each input dimension is
    projected only to C neurons). The non-zero elements are generated randomly
    from a uniform distribution in [-1,1]

    :param M: number of rows
    :param N: number of columns
    :param C: number of nonzero elements
    :return: MxN dense matrix
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
    values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse recurrent kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    take sinput from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1].

    :param M: number of hidden units
    :param C: number of nonzero elements
    :return: MxM dense matrix
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
    values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()

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