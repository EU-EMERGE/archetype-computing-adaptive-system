#ifndef LYAP_DISCRETE_RNN_H
#define LYAP_DISCRETE_RNN_H

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#endif /* LYAP_DISCRETE_RNN_H */
