#include "solve.hpp"

typedef Kokkos::TeamPolicy<>::member_type member_type;

#define RANDOM_LB -2.0
#define RANDOM_UB  2.0

void init_x(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto seed = config.seed;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    if (seed == 0) seed = time(NULL);
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);
    // Kokkos::Random_XorShift64_Pool<> random_pool(52);

    Kokkos::parallel_for(
        "init",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t j) {
            auto generator = random_pool.get_state();
            x(i, j) = generator.drand(RANDOM_LB, RANDOM_UB);
            // x(i, j) = 1.0 / (i+1);
            random_pool.free_state(generator);
        }
    );
}

void evaluate_equations(
    config_t &config, 
    context_t &context, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &f
) {
    auto N = config.N;
    auto order = config.order;
    auto stages = config.stages;
    auto &equations_d = context.equations_d;
    auto &equations_reduce = context.equations_reduce;

    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "equation_prod_evaluate", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {equations_d.total, N}),
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            double prod = 1.0;
            for (int j = 0; j < order; j++) {
                uint8_t index = equations_d.params(i, j);
                // multiply by one 1 if we're out of the param range
                if (index != total_params) prod *= x(index, n);
            }
            equations_reduce(i, n) = prod;
        }
    );

    Kokkos::fence();
    // simple_copy_and_print_2d(red);
    // Kokkos::fence();

    Kokkos::parallel_for(
        "equations_prod_reduce",
        Kokkos::TeamPolicy<>(N * equations_d.sizes.size(), Kokkos::AUTO()),
        KOKKOS_LAMBDA (const member_type &team_member) {
            uint64_t rank = team_member.league_rank();
            uint64_t n = rank / equations_d.sizes.size();
            uint64_t eq = rank % equations_d.sizes.size();

            double sum = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, equations_d.sizes[eq]),
                [&] (uint64_t i, double &lsum) {
                    lsum += equations_reduce(equations_d.indexes[eq] + i, n);
                },
                sum
            );
            f(eq, n) = sum - equations_d.facts[eq];
        }
    );
}

void evaluate_jacobian(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;
    auto order = config.order;
    auto stages = config.stages;

    uint64_t begin[2] = {0, 0};
    uint64_t end[2] = {jacobian_d.total, N};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);
    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "jacobian_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            double prod = 1.0;
            for (int j = 0; j < order; j++) {
                uint8_t index = jacobian_d.params(i, j);
                // if we're out of the param range, we need to multiply bu an integer corresponding 
                // with a total_params offset
                if (index >= total_params) {
                    prod *= index - total_params + 1;
                }
                else {
                    prod *= x(index, n);
                }
            }
            jacobian_reduce(i, n) = prod;
        }
    );

    Kokkos::fence();
    // simple_copy_and_print_2d(red);
    // Kokkos::fence();

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy2({0, 0, 0}, {(uint64_t) total_params, (uint64_t) jacobian_d.sizes.size() / total_params, N});
    Kokkos::parallel_for(
        "jacobian_prod_reduce",
        policy2,
        KOKKOS_LAMBDA (uint64_t derived, uint64_t eq, uint64_t n) {
            uint64_t ind = derived * (jacobian_d.sizes.size() / total_params) + eq;
            double sum = 0;
            for (uint64_t i = 0; i < jacobian_d.sizes[ind]; i++) {
                sum += jacobian_reduce(jacobian_d.indexes[ind] + i, n);
            }
            J(derived, eq, n) = sum;
        }
    );

    // this doesn't really work :(

    // Kokkos::parallel_for(
    //     "jacobian_prod_reduce",
    //     Kokkos::TeamPolicy<>(N * jacobian_d.sizes.size(), Kokkos::AUTO()),
    //     KOKKOS_LAMBDA (const member_type &team_member) {
    //         uint64_t rank = team_member.league_rank();
    //         uint64_t n = rank / jacobian_d.sizes.size();
    //         uint64_t tmp = rank % jacobian_d.sizes.size();
    //         uint64_t eq = tmp / total_params;
    //         uint64_t derived = tmp % total_params;

    //         uint64_t ind = derived * (jacobian_d.sizes.size() / total_params) + eq;

    //         double sum = 0;
    //         Kokkos::parallel_reduce(
    //             Kokkos::TeamThreadRange(team_member, jacobian_d.sizes[ind]),
    //             [&] (uint64_t i, double &lsum) {
    //                 lsum += red(jacobian_d.indexes[ind] + i, n);
    //             },
    //             sum
    //         );
    //         J(derived, eq, n) = sum;
    //     }
    // );

}

void batched_transposed_gemm(config_t &config, context_t &context) {
    using namespace KokkosBatched;
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    // nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare nightmare
    Kokkos::deep_copy(A, 0.0);
    Kokkos::fence();

    Kokkos::parallel_for(
        "batched_transposed_gemm",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _J = Kokkos::subview(J, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _A = Kokkos::subview(A, Kokkos::ALL(), Kokkos::ALL(), n);
            TeamGemm<member_type, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Blocked>::invoke(
                team_member, 1, _J, _J, 1, _A
            );
        }
    );
}

void batched_gemv(config_t &config, context_t &context) {
    using namespace KokkosBatched;
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());

    Kokkos::parallel_for(
        "batched_gemv",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _J = Kokkos::subview(J, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _f = Kokkos::subview(f, Kokkos::ALL(), n);
            auto _b = Kokkos::subview(b, Kokkos::ALL(), n);
            TeamVectorGemv<member_type, Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
                team_member, -1.0, J, f, 0.0, b
            );
        }
    );
}

void batched_gesv(config_t &config, context_t &context) {
    using namespace KokkosBatched;
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());

    Kokkos::parallel_for(
        "batched_gesv",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _A = Kokkos::subview(A, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _b = Kokkos::subview(b, Kokkos::ALL(), n);
            auto _x = Kokkos::subview(dx, Kokkos::ALL(), n);
            TeamVectorGesv<member_type, Gesv::NoPivoting>::invoke(
                team_member, _A, _x, _b 
            );
        }
    );
}

void transpose(Kokkos::View<double ***> &v, Kokkos::View<double ***> &vT) {
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2)});
    Kokkos::parallel_for(
        "transpose",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t j, uint64_t n) {
            vT(j, i, n) = v(i, j, n);
        }
    );
}

void update_weights(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    Kokkos::parallel_for(
        "update_weights",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            // x(i, n) += Kokkos::max(Kokkos::abs(dx(i, n)), bound) * (dx(i, n) / Kokkos::abs(dx(i, n)));
            x(i, n) += dx(i, n) * accept(n);
        }
    );
}

void check_and_swap(config_t &config, context_t &context, double upper_tol) {
    EXPAND_CONTEXT
    auto N = config.N;

    auto t = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(t);

    Kokkos::parallel_for(
        "check_and_swap",
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO()),
        KOKKOS_LAMBDA (const member_type &team_member) {
            uint64_t n = team_member.league_rank();
            ttl(n) += 1;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, x.extent(0)),
                [&] (uint64_t i) {
                    auto generator = random_pool.get_state();
                    if (
                        norms(n) > upper_tol         ||
                        norms(n) < config.accept_tol ||
                        ttl(n) > config.ttl          ||
                        lambdas(n) > 999             ||
                        Kokkos::isnan(f(i, n))
                    ) {
                        x(i, n) = generator.drand(RANDOM_LB, RANDOM_UB);
                        lambdas(n) = 1e-4;
                        ttl(n) = 0;
                        // Kokkos::printf("haha %f %f ", norm, tol);
                    }
                    random_pool.free_state(generator);
                }
            );
            team_member.team_barrier();
        }
    );
}

void batched_norms(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::parallel_for(
        "batched norms",
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO()),
        KOKKOS_LAMBDA (const member_type &team_member) {
            uint64_t n = team_member.league_rank();

            // custom reductor ?
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, f.extent(0)),
                [&] (uint64_t i, double &lnorm) {
                    lnorm += f(i, n) * f(i, n);
                },
               norms(n)
            );
            team_member.team_barrier();
            norms(n) = Kokkos::sqrt(norms(n));
        }
    );
}

void batched_speeds(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::parallel_for(
        "batched speeds",
        Kokkos::RangePolicy(0, N),
        KOKKOS_LAMBDA (uint64_t n) {
            speeds(n) = norms_last(n) - norms(n);
            norms_last(n) = norms(n);
        }
    );
}

void levenberg(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;

    Kokkos::parallel_for(
        "lambda rescale",
        Kokkos::RangePolicy(0, N),
        KOKKOS_LAMBDA (uint64_t n) {
            lambdas(n) *= accept(n) ? 0.7 : 5;
            lambdas(n) = Kokkos::min(lambdas(n), 1e3);
            // lambdas(n) *= speeds(n) < 0 ? 1.5 : 0.2;
            // lambdas(n) = Kokkos::abs(speeds(n) * 100 + lambdas(n)) / 2;
        }
    );

    Kokkos::fence();

    Kokkos::parallel_for(
        "levenberg",
        Kokkos::MDRangePolicy({0, 0}, {A.extent(0), N}),
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            A(i, i, n) += lambdas(n) * A(i, i, n);
        }
    );
}

void backtrack(config_t &config, context_t &context) {
    EXPAND_CONTEXT
    auto N = config.N;
    // Kokkos::deep_copy(alphas, 1e3);

    Kokkos::parallel_for(
        "x + alpha * dx",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {x.extent(0), N}),
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            x_tmp(i, n) = x(i, n) + dx(i, n);
        }
    );
    Kokkos::fence();

    evaluate_equations(config, context, x_tmp, f_tmp);
    Kokkos::fence();

    Kokkos::parallel_for(
        "norms and compare",
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO()),
        KOKKOS_LAMBDA (const member_type &team_member) {
            uint64_t n = team_member.league_rank();

            // custom reductor ?
            double norm_1;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, f.extent(0)),
                [&] (uint64_t i, double &lnorm) {
                    lnorm += f(i, n) * f(i, n);
                },
                norm_1
            );
            team_member.team_barrier();

            double norm_2;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, f.extent(0)),
                [&] (uint64_t i, double &lnorm) {
                    lnorm += f_tmp(i, n) * f_tmp(i, n);
                },
                norm_2
            );
            team_member.team_barrier();

            accept(n) = norm_1 > norm_2;

            team_member.team_barrier();
        }
    );
    Kokkos::fence();
}