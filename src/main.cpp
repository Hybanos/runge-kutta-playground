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

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {

        config_t config;
        context_t context;
        CLI::App app("rkp");
        argv = app.ensure_utf8(argv);

        app.add_flag("--omp");
        app.add_flag("--hip");
        app.add_flag("--cuda");

        app.add_flag_callback("--no-backtrack", [&](){config.use_backtrack = false;});
        app.add_flag_callback("--no-levenberg", [&](){config.use_levenberg = false;});

        app.add_option("-N,-n", config.N, "Number of tableaux to compute in parallel");
        app.add_option("-p,--order", config.order, "Order of the method");
        app.add_option("-s,--stages", config.stages, "Stages of the method");
        app.add_option("-i,--max_iter", config.max_iter, "Maximum number of iterations");
        app.add_option("--checkpoint-save-freq", config.checkpoint_save_freq, "Number of iterations between checkpoint json save");
        app.add_option("--print-freq", config.print_freq, "Number of iterations between norms print");
        app.add_option("--solution-count", config.solution_count, "Number of solutions to find before exiting the program");
        app.add_option("--ttl", config.ttl, "Maximum number of lived iterations before being reset");
        app.add_option("--accept-tol", config.accept_tol, "Residual norm under which the method is accepted");
        app.add_option("--seed", config.seed, "Seed for initial values, 0 for random");

        app.add_flag("--dump-equations", config.dump_equations, "Print equations and Jacobian");
        app.add_flag("--dump-state", config.dump_state, "Print values used while computing");
        app.add_flag("--dump-mem,--dump-memory", config.dump_mem, "Prints memory usage");
        app.add_flag("--wipe", config.wipe_checkpoint, "Clears previously saved checkpoints");

        app.set_help_flag("-h,--help", "?");

        CLI11_PARSE(app, argc, argv);

        // generate trees
        pool p;
        p.gen(config.order);

        uint8_t total_params = (config.stages - 1) * (config.stages - 2) / 2 + config.stages + config.stages - 1;
        auto device_space = Kokkos::DefaultExecutionSpace();
        auto host_space = Kokkos::DefaultHostExecutionSpace();

        // build equation array and jacobian matrix 
        host_equations equations_h = build_equations_or_get_cached(p, config.order, config.stages);
        host_jacobian jacobian_h = build_jacobian_or_get_cached(p, config.order, config.stages, equations_h);

        if (config.dump_equations) {
            print_equations(config.order, config.stages, equations_h);
            print_jacobian(config.order, config.stages, jacobian_h);

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
        context.equations_d = device_equations{
            .params = decltype(device_equations::params)("eq_param_d", equations_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.sizes, "eq_sizes_d"),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.indexes, "eq_indexes_d"),
            .facts = Kokkos::create_mirror_view_and_copy(device_space, equations_h.facts, "eq_facts_d"),
            .total = equations_h.total
        };

        context.jacobian_d = device_jacobian{
            .params = decltype(device_jacobian::params)("jc_param_d", jacobian_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.sizes, "jd_sizes_d"),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.indexes, "jc_indexes_d"),
            .total = jacobian_h.total
        };

        auto tmp_equation_alloc = Kokkos::create_mirror_view(context.equations_d.params);
        Kokkos::deep_copy(tmp_equation_alloc, equations_h.params);
        Kokkos::deep_copy(context.equations_d.params, tmp_equation_alloc);
        
        auto tmp_jacobian_alloc = Kokkos::create_mirror_view(context.jacobian_d.params);
        Kokkos::deep_copy(tmp_jacobian_alloc, jacobian_h.params);
        Kokkos::deep_copy(context.jacobian_d.params, tmp_jacobian_alloc);

        context.x = Kokkos::View<double  **>("x", total_params, config.N);
        context.norms = Kokkos::View<double   *>("norms", config.N);
        context.speeds = Kokkos::View<double   *>("speeds", config.N);

        if (config.wipe_checkpoint) wipe_checkpoint(config.stages);
        if (!load_checkpoint(config.order, config.stages, context.x, context.norms, context.speeds)) init_x(context.x, config.seed);
        Kokkos::fence();
        config.N = context.norms.extent(0);

        context.equations_reduce = Kokkos::View<double  **>("eq_reduce", equations_h.total, config.N);
        context.jacobian_reduce = Kokkos::View<double  **>("jc_reduce", jacobian_h.total, config.N);

        // Kokkos::View<int     **> ipiv("ipiv", total_params, config.N);
        context.f = Kokkos::View<double  **>("f", equations_h.sizes.size(), config.N);
        context.f_tmp = Kokkos::View<double  **>("f_back", equations_h.sizes.size(), config.N);
        context.J = Kokkos::View<double ***>("J", total_params, equations_h.sizes.size(), config.N);
        context.A = Kokkos::View<double ***>("A", total_params, total_params, config.N);
        context.b = Kokkos::View<double  **>("b", total_params, config.N);
        context.dx = Kokkos::View<double  **>("dx", total_params, config.N);
        context.x_tmp = Kokkos::View<double  **>("x_tmp", total_params, config.N);
        context.norms_last = Kokkos::View<double   *>("norms_last", config.N);
        context.alphas = Kokkos::View<double   *>("alphas", config.N);
        context.lambdas = Kokkos::View<double   *>("lambdas", config.N);
        context.accept = Kokkos::View<uint8_t  *>("accept", config.N);
        context.ttl = Kokkos::View<uint64_t *>("ttl", config.N);

        if (config.dump_mem) {
            std::cout << 
                context.equations_d.params.label() << " " << context.equations_d.params.size() * 1 << "\n" <<
                context.equations_d.sizes.label() << " " << context.equations_d.sizes.size() * 4 << "\n" <<
                context.equations_d.indexes.label() << " " << context.equations_d.indexes.size() * 4 << "\n" <<
                context.equations_d.facts.label() << " " << context.equations_d.facts.size() * 8<< "\n" <<
                context.jacobian_d.params.label() << " " << context.jacobian_d.params.size() * 1<< "\n" <<
                context.jacobian_d.sizes.label() << " " << context.jacobian_d.sizes.size() * 4 << "\n" <<
                context.jacobian_d.indexes.label() << " " << context.jacobian_d.indexes.size() * 4 << "\n" <<
                
                context.x.label() << " " << context.x.size() * 8 << "\n" <<
                context.norms.label() << " " << context.norms.size()  * 8 << "\n" <<
                context.speeds.label() << " " << context.speeds.size()  * 8 << "\n" <<

                context.equations_reduce.label() << " " << context.equations_reduce.size()  * 8 << "\n" <<
                context.jacobian_reduce.label() << " " << context.jacobian_reduce.size()  * 8 << "\n" <<

                context.f.label() << " " << context.f.size()  * 8 << "\n" <<
                context.f_tmp.label() << " " << context.f_tmp.size()  * 8 << "\n" <<
                context.J.label() << " " << context.J.size()  * 8 << "\n" <<
                context.A.label() << " " << context.A.size()  * 8 << "\n" <<
                context.b.label() << " " << context.b.size()  * 8 << "\n" <<
                context.dx.label() << " " << context.dx.size()  * 8 << "\n" <<
                context.x_tmp.label() << " " << context.x_tmp.size()  * 8 << "\n" <<
                context.norms_last.label() << " " << context.norms_last.size()  * 8 << "\n" <<
                context.alphas.label() << " " << context.alphas.size()  * 8 << "\n" <<
                context.lambdas.label() << " " << context.lambdas.size()  * 8 << "\n" <<
                context.accept.label() << " " << context.accept.size()  * 1 << "\n" <<
                context.ttl.label() << " " << context.ttl.size()  * 8 << std::endl;
        }

        Kokkos::deep_copy(context.ttl, 0);
        Kokkos::deep_copy(context.alphas, 1e-10);
        Kokkos::deep_copy(context.lambdas, 1e-4);

        for (int i = 0; i < config.max_iter; i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            evaluate_equations(config.N, config.order, config.stages, context.equations_d, context.x, context.equations_reduce, context.f);
            evaluate_jacobian(config.N, config.order, config.stages, context.jacobian_d, context.x, context.jacobian_reduce, context.J);
            Kokkos::fence();

            // compute A = J.T @ J
            batched_transposed_gemm(config.N, context.J, context.A);

            // compute b = -J.T @ f
            for (int n = 0; n < config.N; n++) {
                auto _J = Kokkos::subview(context.J, Kokkos::ALL, Kokkos::ALL, n);
                auto _f = Kokkos::subview(context.f, Kokkos::ALL, n);
                auto _b = Kokkos::subview(context.b, Kokkos::ALL, n);
                KokkosBlas::gemv("N", -1, _J, _f, 0, _b);
            }
            // batched_gemv(N, J, f, b);
            Kokkos::fence();

            // Ghetto-Levenberg-Marquartdt
            Kokkos::deep_copy(context.accept, true);
            if (config.use_levenberg) levenberg(config.N, context.A, context.lambdas, context.speeds, context.accept);

            // solve A @ dx = b for dx
            batched_gesv(config.N, context.A, context.b, context.dx);
            Kokkos::fence();

            if (config.dump_state) {
                simple_copy_and_print_2d(context.x);
                simple_copy_and_print_2d(context.f);
                simple_copy_and_print_3d(context.J);
                simple_copy_and_print_3d(context.A);
                simple_copy_and_print_2d(context.b);
                simple_copy_and_print_2d(context.dx);
            }

            // backtrack
            if (config.use_backtrack) {
                backtrack(config.N, config.order, config.stages, context.equations_d, context.x, context.equations_reduce, context.f, context.f_tmp, context.dx, context.x_tmp, context.alphas, context.accept);
            } else {
                Kokkos::deep_copy(context.alphas, 1);
            }
            Kokkos::fence();

            // update x
            update_weights(context.x, context.dx, context.alphas, context.accept);
            Kokkos::fence();

            batched_norms(config.N, context.f, context.norms);
            batched_speeds(config.N, context.norms, context.norms_last, context.speeds);
            Kokkos::fence();

            if (append_solution(config.N, config.order, config.stages, context.x, context.norms, config.accept_tol) >= config.solution_count) break;
            Kokkos::fence();
            check_and_swap(config.N, context.f, context.x, context.norms, context.alphas, context.lambdas, context.speeds, context.ttl, config.ttl, p.count_trees(), config.accept_tol);
            Kokkos::fence();

            if (!(i%config.print_freq)) {
                
                simple_copy_and_print_1d(context.norms);
                simple_copy_and_print_1d(context.lambdas);
                simple_copy_and_print_1d(context.alphas);

                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << i << " " << 
                          "ips: " << (int) (1.0 / ((t2 - t1).count() / 1e9) * config.N) << 
                          ", " << (t2 - t1).count() / 1e9 << "s" << 
                          std::endl;
            }
            Kokkos::fence();
            if (!(i%config.checkpoint_save_freq)) save_checkpoint(config.N, config.order, config.stages, context.x, context.norms, context.speeds);
            Kokkos::fence();
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    Kokkos::finalize();

    return 0;
}