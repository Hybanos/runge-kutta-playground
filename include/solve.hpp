#pragma once

#include "equations.hpp"

void evaluate_equations(uint32_t N, uint8_t stages, device_equations &equations, Kokkos::View<double **> &x, Kokkos::View<double **> &tmp);