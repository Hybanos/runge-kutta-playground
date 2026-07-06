#include "solve.hpp"

void evaluate_equations(uint32_t N, uint8_t stages, device_equations &equations, Kokkos::View<double **> &x, Kokkos::View<double **> &tmp) {

    uint64_t begin[2] = {0, 0};
    uint64_t end[2] = {N, equations.total};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);

    Kokkos::parallel_for(
        "equation_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t n, uint64_t i) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = equations.params(i, j);
                prod *= x(n, index);
            }
            tmp(n, i) = prod;
        }
    );
}