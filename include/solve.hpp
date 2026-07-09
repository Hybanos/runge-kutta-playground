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
void transpose(Kokkos::View<double ***> &v, Kokkos::View<double ***> &vT);
void update_weights(int N, Kokkos::View<double **> &x, Kokkos::View<double **> &dx);

template<class T>
void simple_copy_and_print_2d(Kokkos::View<T **> &v) {
    auto tmp = Kokkos::create_mirror_view(v);
    auto copy = Kokkos::View<double **, Kokkos::DefaultHostExecutionSpace>("", v.extents());

    Kokkos::deep_copy(tmp, v);
    Kokkos::deep_copy(copy, tmp);

    std::cout << "matrix: " << v.label() << std::endl;
    // print transposed
    for (int j = 0; j < v.extent(1); j++) {
        for (int i = 0; i < v.extent(0); i++) {
            std::cout << copy(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}

template<class T>
void simple_copy_and_print_3d(Kokkos::View<T ***> &v) {
    auto tmp = Kokkos::create_mirror_view(v);
    auto copy = Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace>("", v.extents());

    Kokkos::deep_copy(tmp, v);
    Kokkos::deep_copy(copy, tmp);

    std::cout << "matrix: " << v.label() << std::endl;
    // print transposed
    for (int i = 0; i < v.extent(0); i++) {
        std::cout << "layer n=" << i << std::endl;
        for (int j = 0; j < v.extent(1); j++) {
            for (int k = 0; k < v.extent(2); k++) {
                std::cout << copy(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

template<class T>
void debug_view(Kokkos::View<T> &v) {
    std::cout << "View : " << v.label() << std::endl;
    std::cout << "extents : ";
    for (int i = 0; i < v.rank(); i++) {
        std::cout << v.extent(i) << " ";
    }
    std::cout << std::endl;
}