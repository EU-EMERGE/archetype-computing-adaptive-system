#!/usr/bin/env python3
"""
Python porting of lyap_discreteRNN.c
Computation of Lyapunov Spectrum for a Discrete-Time, Input-Driven RNN

This program computes the Lyapunov spectrum of a discrete-time recurrent
neural network (RNN) driven by an external input, using the standard
discrete-time Benettin algorithm (Benettin et al., 1980) as described in
Pikovsky & Politi (2016), *Lyapunov Exp onents: A Tool to Explore Complex
Dynamics* (Cambridge University Press).

The RNN dynamics is defined as:
    h_{t+1} = tanh(W h_t + V u_t + b)
where: 
    - h_t ∈ ℝ^N is the hidden state,
    - u_t ∈ ℝ^{input_dim} is the external input at time t,
    - W is the recurrent (hidden-to-hidden) weight matrix,
    - V is the input-to-hidden weight matrix,
    - b is the bias vector.

The Jacobian of the system at time t is given by:
    J_t = diag(1 - tanh^2(W h_t + V u_t + b)) · W

The tangent dynamics evolves as:
    δh_{t+1} = J_t δh_t

And periodic QR decomposition of the tangent vectors is used to extract
the Lyapunov exponents from the accumulated logarithmic scaling factors.

References:
- G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn,
  "Lyapunov characteristic exponents for smooth dynamical systems and for
   Hamiltonian systems; A method for computing all of them. Part 1 and 2",
  *Meccanica*, 1980.
- A. Pikovsky & A. Politi (2016),
  *Lyapunov Exponents: A Tool to Explore Complex Dynamics*,
  Cambridge University Press.
"""

import numpy as np
import pandas as pd
from numpy.linalg import qr
from typing import Optional


def load_csv(filename: str) -> np.ndarray:
    """Load CSV data matrix using pandas."""
    return pd.read_csv(filename, header=None).values


def load_vector_csv(filename: str) -> np.ndarray:
    """Load vector from CSV using pandas."""
    return pd.read_csv(filename, header=None).values.flatten()


def compute_lyapunov(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute Lyapunov exponents using the Benettin algorithm with QR decomposition.
    
    Args:
        nl: Number of Lyapunov exponents to compute
        W: Recurrent weight matrix of shape (N, N)
        V: Input weight matrix of shape (N, input_dim)
        b: Bias vector of shape (N,)
        h_traj: Trajectory of hidden states of shape (T, N)
        u_traj: Trajectory of inputs of shape (T, input_dim)
        
    Returns:
        Array of Lyapunov exponents of shape (nl,)
    """
    N = h_traj.shape[1]
    T = h_traj.shape[0]
    input_dim = u_traj.shape[1]
    #print(f"Computing Lyapunov exponents: N={N}, nl={nl}, T={T}, input_dim={input_dim}")
    
    # Initialize tangent vectors randomly and orthonormalize

    V_tan = np.random.randn(N, nl)
    V_tan, _ = qr(V_tan, mode='reduced')
    
    log_norms = np.zeros(nl)
    
    # Main loop over trajectory
    for t in range(T - 1):
        #if t % 100 == 0:
            #print(f"Progress: t={t}/{T}")
        
        # Compute activation derivative: phi'(x) = 1 - tanh^2(x)
        x = W @ h_traj[t] + V @ u_traj[t] + b 
        if fb_traj is not None:
            x += fb_traj[t]
        phi_prime = 1.0 - np.tanh(x) ** 2
        
        # Apply Jacobian: J_t = diag(phi_prime) @ W
        V_tan = phi_prime[:, np.newaxis] * (W @ V_tan)
        
        # Orthonormalize using QR decomposition and accumulate log norms
        Q, R = qr(V_tan, mode='reduced')
        log_norms += np.log(np.abs(np.diag(R)))
        V_tan = Q
    
    #print("Lyapunov computation complete")
    return log_norms / (T - 1)


def main(N: int = 100, T: int = 1000, input_dim: int = 1, nl: Optional[int] = None) -> np.ndarray:
    """
    Main function to load data and compute Lyapunov exponents.
    
    Args:
        N: Dimension of hidden state
        T: Length of trajectory
        input_dim: Dimension of input
        nl: Number of Lyapunov exponents to compute (default: N)
    """
    if nl is None:
        nl = N
    
    print(f"Using N={N}, T={T}, input_dim={input_dim}, nl={nl}")
    
    # Load RNN weights, biases, trajectory and input
    W = load_csv("W.csv")
    V = load_csv("V.csv")
    b = load_vector_csv("b.csv")
    h_traj = load_csv("h_traj.csv")
    u_traj = load_csv("u_timeseries.csv")
    
    # Compute Lyapunov exponents
    lyap = compute_lyapunov(nl, W, V, b, h_traj, u_traj)
    
    # Write results to file
    np.savetxt("lyapunov_rnn.dat", lyap, fmt="%.15e")
    print("Results saved to lyapunov_rnn.dat")
    
    return lyap


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    input_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    nl = int(sys.argv[4]) if len(sys.argv) > 4 else N
    
    if N <= 0 or T <= 0 or input_dim <= 0 or nl <= 0:
        print(f"Usage: {sys.argv[0]} [N] [T] [input_dim] [nl]")
        print(f"Invalid arguments. N={N}, T={T}, input_dim={input_dim}, nl={nl}")
        sys.exit(1)
    
    main(N, T, input_dim, nl)
