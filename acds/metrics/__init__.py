"""
Metrics for analysing reservoir computing networks.

Submodules
----------
attractor
    Attractor dimension metrics (correlation dimension, participation ratio,
    effective rank).
performance
    Performance metrics (NRMSE, etc.).
lyapunov
    Lyapunov exponent computation (C/ctypes with Python fallback).
visualizations
    Plotting utilities for metric analysis.
"""

from .attractor import (
    compute_corr_dim,
    compute_effective_kernel_rank,
    compute_effective_traj_rank,
    compute_participation_ratio,
)
from .lyapunov import compute_lyapunov, compute_lyapunov_from_model
from .performance import nrmse
from .visualizations import (
    heatmap_metrics,
    lineplot_by_n_modules,
    scatter_metrics_vs_performance,
    spider_plot,
)

__all__ = [
    # attractor
    "compute_corr_dim",
    "compute_participation_ratio",
    "compute_effective_traj_rank",
    "compute_effective_kernel_rank",
    # performance
    "nrmse",
    # lyapunov
    "compute_lyapunov",
    "compute_lyapunov_from_model",
    # visualizations
    "scatter_metrics_vs_performance",
    "spider_plot",
    "lineplot_by_n_modules",
    "heatmap_metrics",
]
