/*
--------------------------------------------------------------------------------
Computation of Lyapunov Spectrum for a Discrete-Time, Input-Driven RNN
--------------------------------------------------------------------------------
This program computes the Lyapunov spectrum of a discrete-time recurrent
neural network (RNN) driven by an external input, using the standard
discrete-time Benettin algorithm (Benettin et al., 1980) as described in
Pikovsky & Politi (2016), *Lyapunov Exponents: A Tool to Explore Complex
Dynamics* (Cambridge University Press).

The RNN dynamics is defined as:
    h_{t+1} = tanh(W h_t + V u_t + b),
where:
    - h_t ∈ ℝ^N is the hidden state,
    - u_t ∈ ℝ^{input_dim} is the external input at time t,
    - W is the recurrent (hidden-to-hidden) weight matrix,
    - V is the input-to-hidden weight matrix,
    - b is the bias vector.

The Jacobian of the system at time t is given by:
    J_t = diag(1 - tanh^2(W h_t + V u_t + b)) · W.

The tangent dynamics evolves as:
    δh_{t+1} = J_t δh_t,
and periodic Gram–Schmidt orthonormalization of the tangent vectors is used to
extract the Lyapunov exponents from the accumulated logarithmic stretching
factors.

Implementation details:
- The program does not simulate the RNN dynamics directly.
- Instead, it reconstructs the tangent-space evolution from previously obtained
  quantities that are assumed to be stored in CSV files:
      • the trajectory of hidden states h_t,
      • the corresponding input time series u_t,
      • the recurrent weight matrix W,
      • the input weight matrix V,
      • and the bias vector b.
- Given these, the sequence of Jacobians J_t is computed and used to evolve the
  tangent vectors according to the Benettin algorithm.

References:
- G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn,
  "Lyapunov characteristic exponents for smooth dynamical systems and for
   Hamiltonian systems; A method for computing all of them. Part 1 and 2",
  *Meccanica*, 1980.
- A. Pikovsky & A. Politi (2016),
  *Lyapunov Exponents: A Tool to Explore Complex Dynamics*,
  Cambridge University Press.
--------------------------------------------------------------------------------
*/
#include "lyap_discreteRNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*----------------- INDEXING MACROS ----------------- */
#define W(i,j)      W[(i)*N + (j)]
#define Wf(i,j)     W_f[(i)*N + (j)]
#define V(i,k)      V[(i)*input_dim + (k)]

#define H(t,i)      h_traj[(t)*N + (i)]
#define U(t,k)      u_traj[(t)*input_dim + (k)]

/* ---------------- Gram-Schmidt ---------------- */
void GS(int n, int m, double **V_tan, double *log_norms) {
    int i, j, k;
    for (k = 0; k < m; k++) {
        double norm = 0.0;
        for (i = 0; i < n; i++) norm += V_tan[i][k] * V_tan[i][k];
        norm = sqrt(norm);
        log_norms[k] += log(norm);
        for (i = 0; i < n; i++) V_tan[i][k] /= norm;

        for (j = k + 1; j < m; j++) {
            double dot = 0.0;
            for (i = 0; i < n; i++) dot += V_tan[i][k] * V_tan[i][j];
            for (i = 0; i < n; i++) V_tan[i][j] -= dot * V_tan[i][k];
        }
    }
}

/* ---------------- CSV LOADING ---------------- */
double** load_csv(const char* filename, int rows, int cols) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { printf("Cannot open file %s\n", filename); exit(1); }

    double **data = (double **)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        data[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            if (fscanf(fp, "%lf,", &data[i][j]) != 1) {
                printf("Error reading file %s at row %d col %d\n", filename, i, j);
                exit(1);
            }
        }
    }
    fclose(fp);
    return data;
}

double* load_vector_csv(const char* filename, int n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { printf("Cannot open file %s\n", filename); exit(1); }

    double *vec = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%lf,", &vec[i]) != 1) {
            printf("Error reading vector file %s at index %d\n", filename, i);
            exit(1);
        }
    }
    fclose(fp);
    return vec;
}


void compute_lyapunov(
    int N,
    int nl,
    int T,
    int input_dim,

    const double *W,
    const double *V,
    const double *b,

    const double *h_traj,
    const double *u_traj,
    const double *f_traj,

    double *lyap
) {
    int i, j, k, t; // loop indices
    // Implementation would go here if needed
    /* --- allocate internal work arrays --- */
    double *J = malloc(N * N * sizeof(double));   // Jacobian
    double *Q = malloc(N * N * sizeof(double));   // QR basis
    double *R = malloc(N * N * sizeof(double));   // QR upper-triangular
    double *log_diag = calloc(N, sizeof(double));

    double **V_tan = (double **)malloc(N * sizeof(double*));
    for (i = 0; i < N; i++) {
        V_tan[i] = (double *)malloc(nl * sizeof(double));
        for (j = 0; j < nl; j++)
            V_tan[i][j] = 2.0 * ((double)rand() / RAND_MAX) - 1.0; // random between -1 and 1
    }

    /* --- Allocate log norms --- */
    double *log_norms = (double *)calloc(nl, sizeof(double));

    /* --- Orthonormalize initial tangent vectors --- */
    GS(N, nl, V_tan, log_norms);


    /* --- Main loop over trajectory --- */
    for (t = 0; t < T - 1; t++) {
        double phi_prime[N];

        /* Compute phi'(x) = 1 - tanh^2(W h_t + V u_t + b) */
        for (i = 0; i < N; i++) {
            double x = 0.0;
            for (j = 0; j < N; j++) x += W(i,j) * H(t,j);
            for (j = 0; j < input_dim; j++) x += V(i,j) * U(t,j);
            x += b[i];
            double h_val = tanh(x);
            phi_prime[i] = 1.0 - h_val * h_val;
        }

        /* Multiply tangent vectors by Jacobian J_t = D * W */
        double V_tan_new[N][nl];
        for (i = 0; i < N; i++) {
            for (k = 0; k < nl; k++) {
                double sum = 0.0;
                for (j = 0; j < N; j++) {
                    sum += W(i,j) * V_tan[j][k];
                }
                V_tan_new[i][k] = phi_prime[i] * sum;  // row scaling, correct
            }
        }


        /* Copy back to V_tan */
        for (i = 0; i < N; i++)
            for (k = 0; k < nl; k++)
                V_tan[i][k] = V_tan_new[i][k];

        /* Orthonormalize and accumulate log norms */
        GS(N, nl, V_tan, log_norms);

        /* --- Compute Lyapunov approximations --- */
        for (k = 0; k < nl; k++) {
            lyap[k] = log_norms[k] / (double)(t+1);
        }

    }
}

/* ---------------- MAIN ---------------- */
/*
int main(int argc, char *argv[]) {
    int i, j, k, t;

    // Parse command line arguments for N, T, input_dim
    // Usage: ./lyap_discreteRNN [N] [T] [input_dim]
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) T = atoi(argv[2]);
    if (argc > 3) input_dim = atoi(argv[3]);
    if (N <= 0 || T <= 0 || input_dim <= 0) {
        printf("Invalid arguments. Usage: %s [N] [T] [input_dim]\n", argv[0]);
        exit(1);
    }
    // printf("Using N=%d, T=%d, input_dim=%d\n", N, T, input_dim);

    // --- Load RNN weights, biases, trajectory and input --- 
    W = load_csv("W.csv", N, N);
    W_f = load_csv("W_f.csv", N, N);
    V = load_csv("V.csv", N, input_dim);
    b = load_vector_csv("b.csv", N);
    h_traj = load_csv("h_traj.csv", T, N);
    u_traj = load_csv("u_timeseries.csv", T, input_dim);

    // --- Allocate tangent vectors --- 
    double **V_tan = (double **)malloc(N * sizeof(double*));
    for (i = 0; i < N; i++) {
        V_tan[i] = (double *)malloc(nl * sizeof(double));
        for (j = 0; j < nl; j++)
            V_tan[i][j] = 2.0 * ((double)rand() / RAND_MAX) - 1.0; // random between -1 and 1
    }

    // --- Allocate log norms --- 
    double *log_norms = (double *)calloc(nl, sizeof(double));

    // --- Orthonormalize initial tangent vectors --- 
    GS(N, nl, V_tan, log_norms);

    // --- Open output file --- 
    FILE *lyap_file = fopen("lyapunov_rnn.dat", "w");
    if (!lyap_file) { printf("Cannot open lyapunov_rnn.dat\n"); exit(1); }

    // --- Main loop over trajectory --- 
    for (t = 0; t < T - 1; t++) {
        double phi_prime[N];

        // Compute phi'(x) = 1 - tanh^2(W h_t + V u_t + b)
        for (i = 0; i < N; i++) {
            double x = 0.0;
            for (j = 0; j < N; j++) x += W[i][j] * h_traj[t][j];
            for (j = 0; j < input_dim; j++) x += V[i][j] * u_traj[t][j];
            x += b[i];
            double h_val = tanh(x);
            phi_prime[i] = 1.0 - h_val * h_val;
        }

        // Multiply tangent vectors by Jacobian J_t = D * W 
        double V_tan_new[N][nl];
        for (i = 0; i < N; i++) {
            for (k = 0; k < nl; k++) {
                double sum = 0.0;
                for (j = 0; j < N; j++) {
                    sum += W[i][j] * V_tan[j][k];
                }
                V_tan_new[i][k] = phi_prime[i] * sum;  // row scaling, correct 
            }
        }


        // Copy back to V_tan 
        for (i = 0; i < N; i++)
            for (k = 0; k < nl; k++)
                V_tan[i][k] = V_tan_new[i][k];

        // Orthonormalize and accumulate log norms 
        GS(N, nl, V_tan, log_norms);

        // --- Write Lyapunov approximations to file --- 
        fprintf(lyap_file, "%d", t+1);
        for (k = 0; k < nl; k++) {
            double lyap_approx = log_norms[k] / (double)(t+1);
            fprintf(lyap_file, " %lf", lyap_approx);
        }
        fprintf(lyap_file, "\n");
    }

    // --- Close file --- 
    fclose(lyap_file);




    // --- Free memory --- 
    for (i = 0; i < N; i++) { free(W[i]); free(V[i]); free(V_tan[i]); }
    free(W); free(V); free(b); free(V_tan); free(log_norms);
    for (i = 0; i < T; i++) { free(h_traj[i]); free(u_traj[i]); }
    free(h_traj); free(u_traj);

    return 0;
}

*/