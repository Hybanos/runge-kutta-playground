#include <iostream>

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosLapack_gesv.hpp>

#include "tree.hpp"
#include "combi.hpp"
#include "equations.hpp"
#include "solve.hpp"

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {

        // generate trees
        uint64_t N = 1;
        uint8_t stages = 3;
        uint64_t max_iter = 100;
        if (argc > 1) stages = std::stoi(argv[1]);
        if (argc > 2) N = std::stoi(argv[2]);
        if (argc > 3) max_iter = std::stoi(argv[3]);
        pool p;
        p.gen(stages);

        uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;
        auto device_space = Kokkos::DefaultExecutionSpace();
        auto host_space = Kokkos::DefaultHostExecutionSpace();

        // build equation array and jacobian matrix 
        host_equations equations_h = build_equations_or_get_cached(p, stages);
        host_jacobian jacobian_h = build_jacobian_or_get_cached(p, stages, equations_h);

        // print_equations(stages, equations_h);
        // print_jacobian(stages, jacobian_h);

        // copy
        device_equations equations_d {
            .params = decltype(device_equations::params)("eq_param_d", equations_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.sizes, "eq_sizes_d"),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.indexes, "eq_indexes_d"),
            .facts = Kokkos::create_mirror_view_and_copy(device_space, equations_h.facts, "eq_facts_d"),
            .total = equations_h.total
        };

        device_jacobian jacobian_d {
            .params = decltype(device_jacobian::params)("jc_param_d", jacobian_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.sizes, "jd_sizes_d"),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.indexes, "jc_indexes_d"),
            .total = jacobian_h.total
        };

        // std::cout << "==" << std::endl;
        // for (int i = 0; i < jacobian_h.sizes.size(); i++) {
        //     std::cout << jacobian_h.indexes[i] << "\t" << jacobian_h.sizes[i] << std::endl;
        // }
        // std::cout << jacobian_h.total << std::endl;
        // std::cout << "==" << std::endl;

        auto tmp_equation_alloc = Kokkos::create_mirror_view(equations_d.params);
        Kokkos::deep_copy(tmp_equation_alloc, equations_h.params);
        Kokkos::deep_copy(equations_d.params, tmp_equation_alloc);
        
        auto tmp_jacobian_alloc = Kokkos::create_mirror_view(jacobian_d.params);
        Kokkos::deep_copy(tmp_jacobian_alloc, jacobian_h.params);
        Kokkos::deep_copy(jacobian_d.params, tmp_jacobian_alloc);

        Kokkos::View<double  **> equations_reduce("eq_reduce", equations_h.total, N);
        Kokkos::View<double  **> jacobian_reduce("jc_reduce", jacobian_h.total, N);

        Kokkos::View<double  **> x("x", total_params, N);
        Kokkos::View<int     **> ipiv("ipiv", total_params, N);
        Kokkos::View<double  **> f("f", equations_h.sizes.size(), N);
        Kokkos::View<double  **> f_back("f_back", equations_h.sizes.size(), N);
        Kokkos::View<double ***> J("J", total_params, equations_h.sizes.size(), N);
        Kokkos::View<double ***> A("A", total_params, total_params, N);
        Kokkos::View<double  **> b("b", total_params, N);
        Kokkos::View<double  **> dx("dx", total_params, N);
        Kokkos::View<double **> x_tmp("x_tmp", total_params, N);
        Kokkos::View<double  **> norms("norms", total_params, N);
        Kokkos::View<double   *> alphas("alphas", N);

        init_x(x);
        Kokkos::deep_copy(alphas, 1.0);
        Kokkos::fence();

        for (int i = 0; i < max_iter; i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            evaluate_equations(N, stages, equations_d, x, equations_reduce, f);
            Kokkos::fence();
            evaluate_jacobian(N, stages, jacobian_d, x, jacobian_reduce, J);
            Kokkos::fence();

            // simple_copy_and_print_2d(x);
            // simple_copy_and_print_2d(f);
            // simple_copy_and_print_3d(J);

            // compute A = J.T @ J
            batched_transposed_gemm(N, J, A);
            Kokkos::fence();

            // compute b = -J.T @ f
            for (int n = 0; n < N; n++) {
                auto _J = Kokkos::subview(J, Kokkos::ALL, Kokkos::ALL, n);
                auto _f = Kokkos::subview(f, Kokkos::ALL, n);
                auto _b = Kokkos::subview(b, Kokkos::ALL, n);
                KokkosBlas::gemv("N", -1, _J, _f, 0, _b);
            }
            // batched_gemv(N, J, f, b);
            Kokkos::fence();

            // simple_copy_and_print_3d(A);
            // simple_copy_and_print_2d(b);

            // solve A @ dx = b for dx
            batched_gesv(N, A, b, dx);
            Kokkos::fence();

            // simple_copy_and_print_2d(dx);
            // simple_copy_and_print_2d(b);

            // backtrack
            backtrack(N, stages, equations_d, x, equations_reduce, f, f_back, dx, x_tmp, alphas);

            // update x
            update_weights(x, dx, alphas);
            Kokkos::fence();

            if (!(i%1)) {
                simple_copy_and_print_2d(f);
                simple_copy_and_print_1d(alphas);
                // simple_copy_and_print_2d(x);
                check_and_swap(N, f, x, alphas, p.count_trees());
                Kokkos::fence();
                // simple_copy_and_print_2d(b);
                // simple_copy_and_print_2d(ipiv);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << i << " " << "ips: " << 1.0 / ((t2 - t1).count() / 1e9) * N << std::endl;
            // copy back and print f ?
        }

    }
    Kokkos::finalize();

    return 0;
}