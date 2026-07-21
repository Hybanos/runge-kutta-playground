#pragma once

#include <chrono>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gesv.hpp>
#include <KokkosBatched_Gesv_Impl.hpp>

#include "equations.hpp"

void init_x(Kokkos::View<double **> &x);

void evaluate_equations(
    uint32_t N, uint8_t stages, 
    device_equations &equations_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double **> &f
);

void evaluate_jacobian(
    uint32_t N, uint8_t stages, 
    device_jacobian &jacobian_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double ***> &J
);
void batched_transposed_gemm(uint64_t N, Kokkos::View<double ***> &J, Kokkos::View<double ***> &A);
void batched_gemv(uint64_t N, Kokkos::View<double ***> &J, Kokkos::View<double **> &f, Kokkos::View<double **> &b);
void batched_gesv(uint64_t N, Kokkos::View<double ***> &A, Kokkos::View<double **> &b, Kokkos::View<double **> &x);
void transpose(Kokkos::View<double ***> &v, Kokkos::View<double ***> &vT);
void update_weights(Kokkos::View<double **> &x, Kokkos::View<double **> &dx, Kokkos::View<double *> &alphas);
void check_and_swap(uint64_t N, Kokkos::View<double **> &f, Kokkos::View<double **> &x, Kokkos::View<double *> &alphas, double tol);
void batched_norms(uint64_t N, Kokkos::View<double **> &f, Kokkos::View<double *> &norms);
void backtrack(
    uint64_t N, uint8_t stages, 
    device_equations &equations_d, 
    Kokkos::View<double **> &x,
    Kokkos::View<double **> &red,
    Kokkos::View<double **> &f,
    Kokkos::View<double **> &f_tmp,
    Kokkos::View<double **> &dx,
    Kokkos::View<double **> &x_tmp,
    Kokkos::View<double  *> &alphas
);

template<class T>
void simple_copy_and_print_1d(Kokkos::View<T *> &v) {
    auto tmp = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(tmp, v);

    std::cout << "matrix: " << v.label() << std::endl;
    for (int i = 0; i < v.extent(0); i++) {
        std::cout << tmp(i) << "\t";
    }
    std::cout << std::endl;
}

template<class T>
void simple_copy_and_print_2d(Kokkos::View<T **> &v) {
    auto tmp = Kokkos::create_mirror_view(v);
    auto copy = Kokkos::View<double **, Kokkos::DefaultHostExecutionSpace>("", v.extents());

    Kokkos::deep_copy(tmp, v);
    Kokkos::deep_copy(copy, tmp);

    std::cout << "matrix: " << v.label() << std::endl;
    // print transposed
    for (int i = 0; i < v.extent(0); i++) {
        for (int j = 0; j < v.extent(1); j++) {
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
    for (int n = 0; n < v.extent(2); n++) {
        std::cout << "layer n=" << n << std::endl;
        for (int i = 0; i < v.extent(0); i++) {
            for (int j = 0; j < v.extent(1); j++) {
                std::cout << copy(i, j, n) << "\t";
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