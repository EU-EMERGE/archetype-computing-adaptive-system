"""
Attractor dimension metrics for reservoir computing networks.

Metrics:
- Correlation Dimension (CD)
- Participation Ratio (PR)
- Effective Trajectory Rank
- Effective Kernel Rank
"""

import numpy as np
from skdim.id import CorrInt


def compute_corr_dim(
    trajectory: np.ndarray, k1: int = 5, k2: int = 20, transient: int = 4000
) -> list[float]:
    """Compute Correlation Dimension for each module in a set of trajectories.

    Args:
        trajectory: Trajectory of shape (N_steps, N_modules, N_h).
        k1: First neighborhood size considered.
        k2: Second neighborhood size considered.
        transient: Number of initial steps to discard.

    Returns:
        List of correlation dimension estimates for each module,
        or None if an error occurs.
    """
    corr_dim_values = []
    for i in range(trajectory.shape[1]):
        corr_dim_estimator = CorrInt(k1=k1, k2=k2)
        traj_i = trajectory[transient:, i]
        corr_dim = corr_dim_estimator.fit_transform(traj_i)
        corr_dim_values.append(corr_dim)
    return corr_dim_values


def compute_participation_ratio(
    trajectory: np.ndarray, transient: int = 4000
) -> list[float]:
    """Compute Participation Ratio for each module in a set of trajectories.

    Args:
        trajectory: Trajectory of shape (N_steps, N_modules, N_h).
        transient: Number of initial steps to discard.

    Returns:
        List of participation ratio estimates for each module.
    """
    n_modules = trajectory.shape[1]
    participation_ratios = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        cov_matrix = np.cov(traj_i, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        pr = (np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues**2)
        participation_ratios.append(pr)
    return participation_ratios


def compute_effective_traj_rank(
    trajectory: np.ndarray, transient: int = 4000, eps: float = 1e-10
) -> list[float]:
    """Compute Effective Rank for each module in a set of trajectories.

    Uses SVD-based entropy to estimate the effective dimensionality.

    Args:
        trajectory: Trajectory of shape (N_steps, N_modules, N_h).
        transient: Number of initial steps to discard.
        eps: Small constant to avoid log(0).

    Returns:
        List of effective rank estimates for each module.
    """
    n_modules = trajectory.shape[1]
    ranks = []
    for i in range(n_modules):
        traj_i = trajectory[transient:, i]
        singvals = np.linalg.svd(traj_i, compute_uv=False)
        s = np.sum(np.abs(singvals))
        n_singvals = singvals / s
        entropy = -np.dot(n_singvals + eps, np.log(n_singvals + eps))
        ranks.append(np.exp(entropy))
    return ranks


def compute_effective_kernel_rank(
    trajectory: np.ndarray, eps: float = 1e-10
) -> list[float]:
    """Compute Effective Rank for each module in a set of training states.

    Args:
        trajectory: Trajectory of shape (batch_size, N_steps, N_modules, N_h).
        eps: Small constant to avoid log(0).

    Returns:
        List of effective rank estimates for each module.
    """
    n_modules = trajectory.shape[2]
    ranks = []
    for i in range(n_modules):
        kernel_j = trajectory[:, -1, i]
        singvals = np.linalg.svd(kernel_j, compute_uv=False)
        s = max(np.sum(np.abs(singvals)), eps)
        p = singvals / s
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p + eps))
        ranks.append(np.exp(entropy))
    return ranks


__all__ = [
    "compute_corr_dim",
    "compute_participation_ratio",
    "compute_effective_traj_rank",
    "compute_effective_kernel_rank",
]
