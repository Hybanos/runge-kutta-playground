#pragma once

#include <chrono>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gesv.hpp>
#include <KokkosBatched_Gesv_Impl.hpp>
#include <fmt/format.h>

#include "common.hpp"
#include "equations.hpp"

void init_x(config_t &config, context_t &context);

void eval_generated_equations(
    config_t &config,
    context_t &context,
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &f
);
void eval_generated_jacobian(
    config_t &config,
    context_t &context
);

// void evaluate_equations(config_t &config, context_t &context, Kokkos::View<double **> &x, Kokkos::View<double **> &f);
// void evaluate_jacobian(config_t &config, context_t &context);

void batched_transposed_gemm(config_t &config, context_t &context);
void batched_gemv(config_t &config, context_t &context);
void batched_gesv(config_t &config, context_t &context);
void transpose(Kokkos::View<double ***> &v, Kokkos::View<double ***> &vT);
void update_weights(config_t &config, context_t &context);
void check_and_swap(config_t &config, context_t &context, double upper_tol);
void batched_norms(config_t &config, context_t &context);
void batched_speeds(config_t &config, context_t &context);
void levenberg(config_t &config, context_t &context);
void backtrack(config_t &config, context_t &context);

template<class T>
void simple_copy_and_print_1d(Kokkos::View<T *> &v) {
    auto tmp = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(tmp, v);

    std::cout << "matrix: " << v.label() << std::endl;
    for (int i = 0; i < v.extent(0); i++) {
        std::cout << fmt::format("{:<9.9g}", tmp(i)) << "\t";
        // std::cout << fmt::format("{:6.4g}", tmp(i)) << "\t";
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
            std::cout << fmt::format("{:10.10f}", copy(i, j)) << "\t";
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
                std::cout << fmt::format("{:10.10f}", copy(i, j, n)) << "\t";
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