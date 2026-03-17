"""Quick smoke-test for acds.metrics subpackage.

Run from the repo root:
    python tests/test_metrics.py
"""

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_trajectory(n_steps: int = 5000, n_modules: int = 3, n_hid: int = 16):
    """Return a synthetic trajectory (N_steps, N_modules, N_hid)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_steps, n_modules, n_hid))


def _make_kernel_states(batch: int = 32, n_steps: int = 100,
                        n_modules: int = 3, n_hid: int = 16):
    """Return synthetic training states (batch, N_steps, N_modules, N_hid)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((batch, n_steps, n_modules, n_hid))


# ── tests ────────────────────────────────────────────────────────────────────

def test_imports():
    """All public symbols are importable from acds.metrics."""
    from acds.metrics import (
        compute_corr_dim,
        compute_participation_ratio,
        compute_effective_traj_rank,
        compute_effective_kernel_rank,
        nrmse,
        compute_lyapunov,
        compute_lyapunov_from_model,
        scatter_metrics_vs_performance,
        spider_plot,
        lineplot_by_n_modules,
        heatmap_metrics,
    )
    print("[PASS] all public symbols importable")


def test_nrmse():
    from acds.metrics import nrmse

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert nrmse(a, b) == 0.0, "NRMSE of identical arrays should be 0"

    c = np.array([0.0, 0.0, 0.0])
    val = nrmse(c, a)
    assert val > 0, "NRMSE with different arrays should be > 0"
    print(f"[PASS] nrmse  (identical=0.0, different={val:.4f})")


def test_participation_ratio():
    from acds.metrics import compute_participation_ratio

    traj = _make_trajectory()
    pr = compute_participation_ratio(traj, transient=0)
    assert len(pr) == traj.shape[1]
    assert all(p > 0 for p in pr)
    print(f"[PASS] participation_ratio  PR={[f'{p:.2f}' for p in pr]}")


def test_effective_traj_rank():
    from acds.metrics import compute_effective_traj_rank

    traj = _make_trajectory()
    ranks = compute_effective_traj_rank(traj, transient=0)
    assert len(ranks) == traj.shape[1]
    assert all(r > 0 for r in ranks)
    print(f"[PASS] effective_traj_rank  ranks={[f'{r:.2f}' for r in ranks]}")


def test_effective_kernel_rank():
    from acds.metrics import compute_effective_kernel_rank

    states = _make_kernel_states()
    ranks = compute_effective_kernel_rank(states)
    assert len(ranks) == states.shape[2]
    assert all(r > 0 for r in ranks)
    print(f"[PASS] effective_kernel_rank  ranks={[f'{r:.2f}' for r in ranks]}")


def test_corr_dim():
    from acds.metrics import compute_corr_dim

    traj = _make_trajectory(n_steps=6000)
    cd = compute_corr_dim(traj, transient=1000)
    assert len(cd) == traj.shape[1]
    print(f"[PASS] corr_dim  CD={[f'{c:.2f}' for c in cd]}")


def test_lyapunov_python():
    from acds.metrics import compute_lyapunov

    rng = np.random.default_rng(1)
    N, T, inp_dim, NL = 8, 500, 1, 8

    W = rng.standard_normal((N, N)) * 0.5
    V = rng.standard_normal((N, inp_dim))
    b = np.zeros(N)

    # generate a simple trajectory
    h = np.zeros((T, N))
    u = rng.standard_normal((T, inp_dim))
    for t in range(T - 1):
        h[t + 1] = np.tanh(W @ h[t] + V @ u[t] + b)

    lyap = compute_lyapunov(nl=NL, W=W, V=V, b=b, h_traj=h, u_traj=u, prefer_c=False)
    assert lyap.shape == (NL,)
    print(f"[PASS] lyapunov (python)  top-3={[f'{l:.4f}' for l in lyap]}")


def test_lyapunov_c():
    from acds.metrics import compute_lyapunov

    rng = np.random.default_rng(1)
    N, T, inp_dim, NL = 8, 500, 1, 8

    W = rng.standard_normal((N, N)) * 0.5
    V = rng.standard_normal((N, inp_dim))
    b = np.zeros(N)

    # generate a simple trajectory
    h = np.zeros((T, N))
    u = rng.standard_normal((T, inp_dim))
    for t in range(T - 1):
        h[t + 1] = np.tanh(W @ h[t] + V @ u[t] + b)

    lyap = compute_lyapunov(nl=NL, W=W, V=V, b=b, h_traj=h, u_traj=u, prefer_c=True)
    assert lyap.shape == (NL,)
    print(f"[PASS] lyapunov (C)  top-3={[f'{l:.4f}' for l in lyap]}")


def test_lyapunov_consistency():
    """Check that Python and C implementations give similar results."""
    from acds.metrics import compute_lyapunov

    rng = np.random.default_rng(1)
    N, T, inp_dim, NL = 8, 500, 1, 8

    W = rng.standard_normal((N, N)) * 0.5
    V = rng.standard_normal((N, inp_dim))
    TRIES = 100
    b = np.zeros(N)

    # generate a simple trajectory
    h = np.zeros((T, N))
    u = rng.standard_normal((T, inp_dim))
    for t in range(T - 1):
        h[t + 1] = np.tanh(W @ h[t] + V @ u[t] + b)

    lyap_py, lyap_c = 0, 0
    for _ in range(TRIES):
        lyap_py_ = compute_lyapunov(nl=NL, W=W, V=V, b=b, h_traj=h, u_traj=u, prefer_c=False)
        lyap_c_ = compute_lyapunov(nl=NL, W=W, V=V, b=b, h_traj=h, u_traj=u, prefer_c=True)
        lyap_py += np.array(lyap_py_)
        lyap_c += np.array(lyap_c_)

    lyap_py /= TRIES
    lyap_c /= TRIES

    assert np.allclose(lyap_py, lyap_c, atol=1e-2), f"Python and C Lyapunov exponents differ \nPython: {lyap_py}\nC: {lyap_c}"
    
    print(f"[PASS] lyapunov consistency  max_diff={np.max(np.abs(lyap_py - lyap_c)):.6f}")

# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_imports,
        test_nrmse,
        test_participation_ratio,
        test_effective_traj_rank,
        test_effective_kernel_rank,
        test_corr_dim,
        test_lyapunov_python,
        test_lyapunov_c,
        test_lyapunov_consistency,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            failures += 1

    print(f"\n{'='*50}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    if failures:
        raise SystemExit(1)
