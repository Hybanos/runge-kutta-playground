#include <iostream>
#include <filesystem>

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosLapack_gesv.hpp>
#include <CLI/CLI.hpp>
#include <thread>

#include "tree.hpp"
#include "combi.hpp"
#include "equations.hpp"
#include "solve.hpp"
#include "io.hpp"

struct config_t {
    uint64_t N = 0;
    uint8_t stages = 3;
    uint64_t max_iter = -1;
    uint64_t checkpoint_save_freq = 100;
    uint64_t print_freq = 1;
    uint64_t solution_count = 0;
    uint64_t ttl = -1;
    double accept_tol = 1e-12;

    bool dump_equations = false;
    bool dump_state = false;

    bool wipe_checkpoint = false;
};

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {

        config_t c;
        CLI::App app("rkp");
        argv = app.ensure_utf8(argv);

        app.add_flag("--omp");
        app.add_flag("--hip");
        app.add_flag("--cuda");

        app.add_option("-N,-n", c.N, "Number of tableaux to compute in parallel");
        app.add_option("-s,--stages", c.stages, "Stages of the method");
        app.add_option("-i,--max_iter", c.max_iter, "Maximum number of iterations");
        app.add_option("--checkpoint-save-freq", c.checkpoint_save_freq, "Number of iterations between checkpoint json save");
        app.add_option("--print-freq", c.print_freq, "Number of iterations between norms print");
        app.add_option("--solution-count", c.solution_count, "Number of solutions to find before exiting the program");
        app.add_option("--ttl", c.ttl, "Maximum number of lived iterations before being reset");
        app.add_option("--accept-tol", c.accept_tol, "Residual norm under which the method is accepted");

        app.add_flag("--dump-equations", c.dump_equations, "Print equations and Jacobian");
        app.add_flag("--dump-state", c.dump_state, "Print values used while computing");
        app.add_flag("--wipe", c.wipe_checkpoint, "Clears previously saved checkpoints");

        app.set_help_flag("-h,--help", "?");

        CLI11_PARSE(app, argc, argv);

        // generate trees
        pool p;
        p.gen(c.stages);

        uint8_t total_params = (c.stages - 1) * (c.stages - 2) / 2 + c.stages + c.stages - 1;
        auto device_space = Kokkos::DefaultExecutionSpace();
        auto host_space = Kokkos::DefaultHostExecutionSpace();

        // build equation array and jacobian matrix 
        host_equations equations_h = build_equations_or_get_cached(p, c.stages);
        host_jacobian jacobian_h = build_jacobian_or_get_cached(p, c.stages, equations_h);

        if (c.dump_equations) {
            print_equations(c.stages, equations_h);
            print_jacobian(c.stages, jacobian_h);

            std::cout << "= Equations (index, size) =" << std::endl;
            for (int i = 0; i < equations_h.sizes.size(); i++) {
                std::cout << equations_h.indexes[i] << "\t" << equations_h.sizes[i] << std::endl;
            }
            std::cout << equations_h.total << std::endl;
            std::cout << "==" << std::endl;
            std::cout << "= Jacobian (index, size) =" << std::endl;
            for (int i = 0; i < jacobian_h.sizes.size(); i++) {
                std::cout << jacobian_h.indexes[i] << "\t" << jacobian_h.sizes[i] << std::endl;
            }
            std::cout << jacobian_h.total << std::endl;
            std::cout << "==" << std::endl;
        }

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

        auto tmp_equation_alloc = Kokkos::create_mirror_view(equations_d.params);
        Kokkos::deep_copy(tmp_equation_alloc, equations_h.params);
        Kokkos::deep_copy(equations_d.params, tmp_equation_alloc);
        
        auto tmp_jacobian_alloc = Kokkos::create_mirror_view(jacobian_d.params);
        Kokkos::deep_copy(tmp_jacobian_alloc, jacobian_h.params);
        Kokkos::deep_copy(jacobian_d.params, tmp_jacobian_alloc);

        Kokkos::View<double  **> x("x", total_params, c.N);
        Kokkos::View<double   *> norms("norms", c.N);
        Kokkos::View<double   *> speeds("speeds", c.N);

        if (c.wipe_checkpoint) wipe_checkpoint(c.stages);
        if (!load_checkpoint(c.stages, x, norms, speeds)) init_x(x);
        Kokkos::fence();
        c.N = norms.extent(0);

        Kokkos::View<double  **> equations_reduce("eq_reduce", equations_h.total, c.N);
        Kokkos::View<double  **> jacobian_reduce("jc_reduce", jacobian_h.total, c.N);

        Kokkos::View<int     **> ipiv("ipiv", total_params, c.N);
        Kokkos::View<double  **> f("f", equations_h.sizes.size(), c.N);
        Kokkos::View<double  **> f_back("f_back", equations_h.sizes.size(), c.N);
        Kokkos::View<double ***> J("J", total_params, equations_h.sizes.size(), c.N);
        Kokkos::View<double ***> A("A", total_params, total_params, c.N);
        Kokkos::View<double  **> b("b", total_params, c.N);
        Kokkos::View<double  **> dx("dx", total_params, c.N);
        Kokkos::View<double  **> x_tmp("x_tmp", total_params, c.N);
        Kokkos::View<double   *> norms_last("norms_last", c.N);
        Kokkos::View<double   *> alphas("alphas", c.N);
        Kokkos::View<double   *> lambdas("lambdas", c.N);
        Kokkos::View<uint8_t  *> accept("accept", c.N);
        Kokkos::View<uint64_t *> ttl("ttl", c.N);

        Kokkos::deep_copy(ttl, 0);
        Kokkos::deep_copy(alphas, 1e-10);
        Kokkos::deep_copy(lambdas, 1e-4);

        for (int i = 0; i < c.max_iter; i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            evaluate_equations(c.N, c.stages, equations_d, x, equations_reduce, f);
            evaluate_jacobian(c.N, c.stages, jacobian_d, x, jacobian_reduce, J);
            Kokkos::fence();

            // compute A = J.T @ J
            batched_transposed_gemm(c.N, J, A);

            // compute b = -J.T @ f
            for (int n = 0; n < c.N; n++) {
                auto _J = Kokkos::subview(J, Kokkos::ALL, Kokkos::ALL, n);
                auto _f = Kokkos::subview(f, Kokkos::ALL, n);
                auto _b = Kokkos::subview(b, Kokkos::ALL, n);
                KokkosBlas::gemv("N", -1, _J, _f, 0, _b);
            }
            // batched_gemv(N, J, f, b);
            Kokkos::fence();

            // Ghetto-Levenberg-Marquartdt
            levenberg(c.N, A, lambdas, speeds);

            // solve A @ dx = b for dx
            batched_gesv(c.N, A, b, dx);
            Kokkos::fence();

            if (c.dump_state) {
                simple_copy_and_print_2d(x);
                simple_copy_and_print_2d(f);
                simple_copy_and_print_3d(J);
                simple_copy_and_print_3d(A);
                simple_copy_and_print_2d(b);
                simple_copy_and_print_2d(dx);
            }

            // backtrack
            Kokkos::deep_copy(accept, 1.0);
            backtrack(c.N, c.stages, equations_d, x, equations_reduce, f, f_back, dx, x_tmp, alphas, accept);
            Kokkos::fence();
            // Kokkos::deep_copy(alphas, 1);

            // update x
            update_weights(x, dx, alphas, accept);
            Kokkos::fence();

            batched_norms(c.N, f, norms);
            batched_speeds(c.N, norms, norms_last, speeds);
            Kokkos::fence();

            if (append_solution(c.N, c.stages, x, norms, c.accept_tol) >= c.solution_count) break;
            Kokkos::fence();
            check_and_swap(c.N, f, x, norms, alphas, speeds, ttl, c.ttl, p.count_trees(), c.accept_tol);
            Kokkos::fence();

            if (!(i%c.print_freq)) {
                
                simple_copy_and_print_1d(norms);
                simple_copy_and_print_1d(lambdas);
                simple_copy_and_print_1d(alphas);

                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << i << " " << "ips: " << (int) (1.0 / ((t2 - t1).count() / 1e9) * c.N) << std::endl;
            }
            Kokkos::fence();
            if (!(i%c.checkpoint_save_freq)) save_checkpoint(c.N, c.stages, x, norms, speeds);
            Kokkos::fence();
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    Kokkos::finalize();

    return 0;
}