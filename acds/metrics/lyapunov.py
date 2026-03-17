"""
Lyapunov exponent computation for discrete-time, input-driven RNNs.

Provides a unified ``compute_lyapunov`` that attempts the fast C/ctypes
implementation first, falling back to a pure-Python/NumPy version when the
shared library is unavailable.

References:
    G. Benettin et al., "Lyapunov characteristic exponents …", Meccanica, 1980.
    A. Pikovsky & A. Politi, *Lyapunov Exponents*, Cambridge Univ. Press, 2016.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import numpy as np
from numpy.linalg import qr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the C shared library (best-effort, non-fatal)
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(os.path.dirname(__file__), "lyapunov_c")
_LIBLYAPUNOV: Optional[ctypes.CDLL] = None

for _name in ("liblyapunov.so", "liblyapunov.dylib"):
    _path = os.path.join(_LIB_DIR, _name)
    if os.path.isfile(_path):
        try:
            _LIBLYAPUNOV = ctypes.CDLL(_path)
        except OSError:
            pass
        break

# Also try system-wide (original behaviour of metrics.py)
if _LIBLYAPUNOV is None:
    try:
        _LIBLYAPUNOV = ctypes.CDLL("liblyapunov.so")
    except OSError:
        pass

if _LIBLYAPUNOV is None:
    logger.info(
        "C Lyapunov library not found; will use pure-Python fallback."
    )


# ---------------------------------------------------------------------------
# Pure-Python implementation (ported from lyap_discreteRNN.py)
# ---------------------------------------------------------------------------

def _compute_lyapunov_python(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Benettin algorithm with QR decomposition (pure NumPy).

    Args:
        nl: Number of Lyapunov exponents to compute.
        W: Recurrent weight matrix (N, N).
        V: Input weight matrix (N, input_dim).
        b: Bias vector (N,).
        h_traj: Hidden-state trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    N = h_traj.shape[1]
    T = h_traj.shape[0]

    V_tan = np.random.randn(N, nl)
    V_tan, _ = qr(V_tan, mode="reduced")

    log_norms = np.zeros(nl)

    for t in range(T - 1):
        x = W @ h_traj[t] + V @ u_traj[t] + b
        if fb_traj is not None:
            x += fb_traj[t]
        phi_prime = 1.0 - np.tanh(x) ** 2
        V_tan = phi_prime[:, np.newaxis] * (W @ V_tan)
        Q, R = qr(V_tan, mode="reduced")
        log_norms += np.log(np.abs(np.diag(R)))
        V_tan = Q

    return log_norms / (T - 1)


# ---------------------------------------------------------------------------
# C/ctypes implementation
# ---------------------------------------------------------------------------

def _compute_lyapunov_c(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: np.ndarray,
) -> np.ndarray:
    """Compute Lyapunov exponents using the compiled C library via ctypes.

    Raises ``RuntimeError`` if the C library is not available.
    """
    if _LIBLYAPUNOV is None:
        raise RuntimeError("C Lyapunov library is not loaded.")

    N = h_traj.shape[1]
    n_steps = h_traj.shape[0]
    input_dim = u_traj.shape[1] if u_traj.ndim > 1 else 1

    W_c = np.ascontiguousarray(W, dtype=np.float64)
    V_c = np.ascontiguousarray(V, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    h_c = np.ascontiguousarray(h_traj, dtype=np.float64)
    u_c = np.ascontiguousarray(u_traj, dtype=np.float64)
    fb_c = np.ascontiguousarray(fb_traj, dtype=np.float64)

    lyap_exponents = (ctypes.c_double * nl)()

    _LIBLYAPUNOV.compute_lyapunov(
        N,
        nl,
        n_steps,
        input_dim,
        W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        V_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fb_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lyap_exponents,
    )

    return np.array([lyap_exponents[i] for i in range(nl)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_lyapunov(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
    *,
    prefer_c: bool = True,
) -> np.ndarray:
    """Compute Lyapunov exponents for a discrete-time input-driven RNN.

    Attempts the C implementation first (faster), falling back to pure Python
    if the shared library is unavailable or ``prefer_c`` is ``False``.

    Args:
        nl: Number of Lyapunov exponents to compute.
        W: Recurrent weight matrix (N, N).
        V: Input weight matrix (N, input_dim).
        b: Bias vector (N,).
        h_traj: Hidden-state trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).
        prefer_c: If True (default), use C library when available.

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    if prefer_c and _LIBLYAPUNOV is not None and fb_traj is not None:
        try:
            return _compute_lyapunov_c(nl, W, V, b, h_traj, u_traj, fb_traj)
        except Exception:
            logger.warning(
                "C Lyapunov computation failed; falling back to Python.",
                exc_info=True,
            )

    return _compute_lyapunov_python(nl, W, V, b, h_traj, u_traj, fb_traj)


def compute_lyapunov_from_model(
    model,
    trajectory: np.ndarray,
    inputs: np.ndarray,
    feedbacks: np.ndarray,
    n_lyap: int,
    transient: int = 4000,
) -> list[list[float]]:
    """Compute Lyapunov exponents for every module in an ArchetipesNetwork.

    This is a convenience wrapper that extracts per-module weights and
    trajectories from the model and delegates to :func:`compute_lyapunov`.

    Args:
        model: An ``ArchetipesNetwork`` instance.
        trajectory: Trajectory of shape (N_steps, N_modules, N_h).
        inputs: Inputs of shape (N_steps, N_inp) — shared across modules.
        feedbacks: Feedbacks of shape (N_steps, N_modules, N_h).
        n_lyap: Number of Lyapunov exponents to compute.
        transient: Number of initial steps to discard.

    Returns:
        List of arrays, one per module, each of shape (n_lyap,).
    """
    from acds.archetypes import InterconnectionRON
    from acds.networks.utils import unstack_state

    exponents = []
    for module_idx, module in enumerate(
        unstack_state(model.archetipes_params, model.archetipes_buffers)
    ):
        params_i, buffers_i = module
        ron = InterconnectionRON(model.n_inp, model.n_hid, dt=1.0, gamma=1.0, epsilon=1.0)
        ron.load_state_dict({**params_i, **buffers_i})

        W = ron.h2h.detach().numpy()
        V = ron.x2h.detach().numpy()
        b = ron.bias.detach().numpy()

        trajectory_i = trajectory[:, module_idx, :]
        feedbacks_i = feedbacks[:, module_idx, :]

        exp = compute_lyapunov(
            n_lyap, W, V, b, trajectory_i, inputs, feedbacks_i
        )
        exponents.append(exp.tolist())

    return exponents


__all__ = ["compute_lyapunov", "compute_lyapunov_from_model"]
