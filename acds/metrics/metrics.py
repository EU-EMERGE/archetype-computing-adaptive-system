"""
Code to compute the attractor dimension metrics in the paper ---
List of metrics considered:
- Correlation dimension (CD)
- Kernel rank (KR) i.e. effective rank of the kernel matrix of the trajectory
- Participation ratio (PR)
- Maximum Lyapunov exponent (MLE)
"""

import ctypes
import numpy as np
from skdim.id import CorrInt
from acds.networks import ArchetipesNetwork
from acds.archetypes import InterconnectionRON
from acds.networks.utils import unstack_state

def compute_corr_dim(trajectory: np.ndarray, k1=5,  k2=20, transient=4000) -> list[float]:
    """Compute Correlation Dimension for each module in a set of trajectories

    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        k1 (int, optional): k1 :  First neighborhood size considered. Defaults to 5.
        k2 (int, optional): k2 :  Second neighborhood size considered. Defaults to 20.
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.

    Returns:
        Optional[list[float]]: List of correlation dimension estimates for each module, or None if an error occurs
    """
    corr_dim_values = []
    for i in range(trajectory.shape[1]): # for each module in the network
        corr_dim_estimator = CorrInt(k1=k1, k2=k2)
        traj_i = trajectory[transient:, i]
        corr_dim = corr_dim_estimator.fit_transform(traj_i)
        corr_dim_values.append(corr_dim)

    return corr_dim_values


def compute_participation_ratio(trajectory, transient=4000) -> list[float]:
    """
    Compute Participation Ratio for each module in a set of trajectories
    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.
    Returns:
        Optional[list[float]]: List of participation ratio estimates for each module, or None if an error occurs
    """
    n_modules = trajectory.shape[1]
    participation_ratios = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        # compute covariance matrix
        cov_matrix = np.cov(traj_i, rowvar=False)
        # compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        # compute participation ratio
        pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
        participation_ratios.append(pr)
    return participation_ratios


def compute_effective_rank(trajectory, transient=4000, eps = 1e-10) -> list[float]:
    """
    Compute Effective Rank for each module in a set of trajectories
    Args:
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        transient (int, optional): transient :  Number of initial steps to discard. Defaults to 4000.
    Returns:
        Optional[list[float]]: List of effective rank estimates for each module, or None if an error occurs
    """
    n_modules = trajectory.shape[1]
    ranks = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        singvals = np.linalg.svdvals(traj_i)
        s = np.sum(np.abs(singvals))
        n_singvals = singvals / s
        entropy = - np.dot(n_singvals + eps, np.log(n_singvals + eps)) 
        ranks.append(np.exp(entropy))
    return ranks


def nrmse(preds: np.ndarray, target: np.ndarray) -> float:
    assert preds.shape == target.shape, "Predictions and target must have the same shape"
    mse = np.mean(np.square(preds - target))
    norm = np.sqrt(np.mean(np.square(target)))
    # rmse / norm
    return np.sqrt(mse) / (norm + 1e-9)



def compute_lyapunov(model:ArchetipesNetwork, trajectory: np.ndarray, inputs: np.ndarray, feedbacks: np.ndarray, n_lyap: int, transient: int = 4000) -> list[float]:
    """
    Compute the Lyapunov exponents for a given trajectory using the C library.
    Args:
        model (ArchetipesNetwork): The model used to generate the trajectory
        trajectory (np.ndarray): trajectory of shape (N_steps, N_modules, N_h)
        inputs (np.ndarray): inputs of shape (N_steps, N_inp) - SHARED across all modules
        feedbacks (np.ndarray): feedbacks of shape (N_steps, N_modules, N_h)
        n_lyap (int): number of Lyapunov exponents to compute
        transient (int, optional): number of initial steps to discard. Defaults to 4000.

    Returns:
        list[float]: List of Lyapunov exponents for each module
    """
    from acds.networks.connection_matrices import cycle_matrix
    lib = ctypes.CDLL("liblyapunov.so")
    exponents = []
    for module_idx, module in enumerate(unstack_state(model.archetipes_params, model.archetipes_buffers)):
        params_i, buffers_i = module
        # Prepare parameters and buffers for C function
        ron = InterconnectionRON(model.n_inp, model.n_hid, dt=1.0, gamma=1.0, epsilon=1.0)
        ron.load_state_dict({**params_i, **buffers_i})

        W = np.ascontiguousarray(ron.h2h.detach().numpy())
        V = np.ascontiguousarray(ron.x2h.detach().numpy())
        b = np.ascontiguousarray(ron.bias.detach().numpy())

        # Extract trajectory for this specific module
        trajectory_i = np.ascontiguousarray(trajectory[:, module_idx, :])
        # Inputs are shared across all modules (not indexed by module_idx)
        inputs_i = np.ascontiguousarray(inputs)
        # Extract feedbacks for this specific module  
        feedbacks_i = np.ascontiguousarray(feedbacks[:, module_idx, :])


        W_ctypes = W.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        V_ctypes = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_ctypes = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        traj_ctypes = trajectory_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        inp_ctypes = inputs_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        fb_ctypes = feedbacks_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        n_steps = trajectory_i.shape[0]

        lyap_exponents = (ctypes.c_double * n_lyap)()

        lib.compute_lyapunov(

            model.n_hid,
            n_lyap,
            n_steps,
            model.n_inp,

            W_ctypes, 
            V_ctypes, 
            b_ctypes, 

            traj_ctypes, 
            inp_ctypes, 
            fb_ctypes,
            
            lyap_exponents
        )

        exponents.append([lyap_exponents[i] for i in range(n_lyap)])
    return exponents


__all__ = ['compute_corr_dim', 'compute_participation_ratio', 'compute_effective_rank', 'nrmse']

if __name__ == '__main__':
    # simple test
    from acds.archetypes import InterconnectionRON
    import torch
    from acds.networks.connection_matrices import cycle_matrix 
    inputs = torch.ones((10000, 1))

    model = ArchetipesNetwork([InterconnectionRON(1, 2, 1, 1, 1, rho=0.9) for _ in range(4)], cycle_matrix(4))
    
    traj, fbs = model(inputs)
    traj = traj.detach().numpy()
    inputs = inputs
    fbs = fbs.detach().numpy()
    print("Correlation Dimension:", compute_corr_dim(traj))
    print("Participation Ratio:", compute_participation_ratio(traj))
    print("Effective Rank:", compute_effective_rank(traj))
    print("Lyapunov Exponents:", compute_lyapunov(model, traj, inputs, fbs, n_lyap=1))

