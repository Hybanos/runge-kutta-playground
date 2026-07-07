#pragma once

#include "equations.hpp"

void init_x(Kokkos::View<double **> &x);

void evaluate_equations(
    uint32_t N, uint8_t stages, 
    host_equations &equations_h, 
    device_equations &equations_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double **> &f
);

void evaluate_jacobian(
    uint32_t N, uint8_t stages, 
    host_jacobian &jacobian_h, 
    device_jacobian &jacobian_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double ***> &J
);

void simple_copy_and_print_2d(Kokkos::View<double **> &v);
void simple_copy_and_print_3d(Kokkos::View<double ***> &v);