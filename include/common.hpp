#pragma once

#include <Kokkos_Core.hpp>

#include "equations.hpp"

struct config_t {
    uint64_t N = 0;
    uint8_t stages = 3;
    uint8_t order = 3;
    uint64_t max_iter = -1;
    uint64_t checkpoint_save_freq = 100;
    uint64_t print_freq = 1;
    uint64_t solution_count = -1;
    uint64_t ttl = -1;
    long seed = 0;

    double accept_tol = 1e-12;

    bool use_backtrack = true;
    bool use_levenberg = true;

    bool dump_equations = false;
    bool dump_state = false;
    bool dump_mem = false;

    bool wipe_checkpoint = false;
};

struct context_t {
    device_equations equations_d;
    device_jacobian jacobian_d;

    Kokkos::View<double  **> x;
    Kokkos::View<double   *> norms;
    Kokkos::View<double   *> speeds;

    Kokkos::View<double  **> equations_reduce;
    Kokkos::View<double  **> jacobian_reduce;

    // Kokkos::View<int     **> ipiv;
    Kokkos::View<double  **> f;
    Kokkos::View<double  **> f_tmp;
    Kokkos::View<double ***> J;
    Kokkos::View<double ***> A;
    Kokkos::View<double  **> b;
    Kokkos::View<double  **> dx;
    Kokkos::View<double  **> x_tmp;
    Kokkos::View<double   *> norms_last;
    Kokkos::View<double   *> alphas;
    Kokkos::View<double   *> lambdas;
    Kokkos::View<uint8_t  *> accept;
    Kokkos::View<uint64_t *> ttl;
};

// waiting for reflection
#define EXPAND_CONTEXT \
device_equations &equations_d = context.equations_d; \
device_jacobian &jacobian_d = context.jacobian_d; \
 \
Kokkos::View<double  **> &x = context.x; \
Kokkos::View<double   *> &norms = context.norms; \
Kokkos::View<double   *> &speeds = context.speeds; \
 \
Kokkos::View<double  **> &equations_reduce = context.equations_reduce; \
Kokkos::View<double  **> &jacobian_reduce = context.jacobian_reduce; \
 \
Kokkos::View<double  **> &f = context.f; \
Kokkos::View<double  **> &f_tmp = context.f_tmp; \
Kokkos::View<double ***> &J = context.J; \
Kokkos::View<double ***> &A = context.A; \
Kokkos::View<double  **> &b = context.b; \
Kokkos::View<double  **> &dx = context.dx; \
Kokkos::View<double  **> &x_tmp = context.x_tmp; \
Kokkos::View<double   *> &norms_last = context.norms_last; \
Kokkos::View<double   *> &alphas = context.alphas; \
Kokkos::View<double   *> &lambdas = context.lambdas; \
Kokkos::View<uint8_t  *> &accept = context.accept; \
Kokkos::View<uint64_t *> &ttl = context.ttl;
