#include "solve.hpp"

typedef Kokkos::TeamPolicy<>::member_type member_type;

void init_x(Kokkos::View<double **> &x) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    // Kokkos::Random_XorShift64_Pool<> random_pool(time(NULL));
    Kokkos::Random_XorShift64_Pool<> random_pool(2);

    Kokkos::parallel_for(
        "init",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t j) {
            auto generator = random_pool.get_state();
            x(i, j) = generator.drand(-2.0, 2.0);
            // x(i, j) = 0.1 * (i+1);
            random_pool.free_state(generator);
        }
    );
}

void evaluate_equations(    
    uint32_t N, uint8_t stages, 
    host_equations &equations_h, 
    device_equations &equations_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double **> &f) {

    uint64_t begin[2] = {0, 0};
    uint64_t end[2] = {equations_h.total, N};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);
    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "equation_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = equations_d.params(i, j);
                // multiply by one 1 if we're out of the param range
                if (index != total_params) prod *= x(index, n);
            }
            red(i, n) = prod;
        }
    );

    Kokkos::fence();
    // simple_copy_and_print_2d(red);
    // Kokkos::fence();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy2({0, 0}, {(int) equations_h.sizes.size(), N});
    Kokkos::parallel_for(
        "equation_prod_reduce",
        policy2,
        KOKKOS_LAMBDA (uint64_t eq, uint64_t n) {
            double sum = 0;
                for (uint64_t i = 0; i < equations_d.sizes[eq]; i++) {
                    sum += red(equations_d.indexes[eq] + i, n);
                }
                f(eq, n) = sum - equations_d.facts[eq];
        }
    );
    Kokkos::fence();
}

void evaluate_jacobian(
    uint32_t N, uint8_t stages, 
    host_jacobian &jacobian_h, 
    device_jacobian &jacobian_d, 
    Kokkos::View<double **> &x, 
    Kokkos::View<double **> &red, 
    Kokkos::View<double ***> &J) {

    uint64_t begin[2] = {0, 0};
    uint64_t end[2] = {jacobian_h.total, N};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);
    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "jacobian_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = jacobian_d.params(i, j);
                // multiply by one if we're out of the param range
                if (index != total_params) prod *= x(index, n);
            }
            red(i, n) = prod;
        }
    );

    Kokkos::fence();
    // simple_copy_and_print_2d(red);
    // Kokkos::fence();

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy2({0, 0, 0}, {(int) total_params, (int) jacobian_h.sizes.size() / total_params, N});
    Kokkos::parallel_for(
        "jacobian_prod_reduce",
        policy2,
        KOKKOS_LAMBDA (uint64_t derived, uint64_t eq, uint64_t n) {
            uint64_t ind = derived * (jacobian_h.sizes.size() / total_params) + eq;
            double sum = 0;
            for (uint64_t i = 0; i < jacobian_d.sizes[ind]; i++) {
                sum += red(jacobian_d.indexes[ind] + i, n);
            }
            J(derived, eq, n) = sum;
        }
    );

    Kokkos::fence();
}

void batched_transposed_gemm(uint64_t N, Kokkos::View<double ***> &J, Kokkos::View<double ***> &A) {
    using namespace KokkosBatched;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());

    Kokkos::parallel_for(
        "batched_transposed_gemm",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _J = Kokkos::subview(J, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _A = Kokkos::subview(A, Kokkos::ALL(), Kokkos::ALL(), n);
            TeamGemm<member_type, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>::invoke(
                team_member, 1, _J, _J, 1, _A
            );
        }
    );
    Kokkos::fence();
}

void batched_gemv(uint64_t N, Kokkos::View<double ***> &J, Kokkos::View<double **> &f, Kokkos::View<double **> &b) {
    using namespace KokkosBatched;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());

    Kokkos::parallel_for(
        "batched_gemv",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _J = Kokkos::subview(J, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _f = Kokkos::subview(f, Kokkos::ALL(), n);
            auto _b = Kokkos::subview(b, Kokkos::ALL(), n);
            TeamGemv<member_type, Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
                team_member, -1, _J, _f, 0, _b
            );
        }
    );
    Kokkos::fence();
}

void batched_gesv(uint64_t N, Kokkos::View<double ***> &A, Kokkos::View<double **> &b, Kokkos::View<int **> &ipiv) {
    using namespace KokkosBatched;

    Kokkos::TeamPolicy<> policy(N, Kokkos::AUTO());

    Kokkos::parallel_for(
        "batched_gesv",
        policy,
        KOKKOS_LAMBDA (member_type team_member) {
            uint64_t n = team_member.league_rank();
            auto _A = Kokkos::subview(A, Kokkos::ALL(), Kokkos::ALL(), n);
            auto _b = Kokkos::subview(b, Kokkos::ALL(), n);
            auto _ipiv = Kokkos::subview(ipiv, Kokkos::ALL(), n);
            TeamGesv<member_type, Algo::Gemv::Unblocked>::invoke(
                team_member, _A, _b, _b
            );
        }
    );
    Kokkos::fence();
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
    Kokkos::fence();
}

void update_weights(Kokkos::View<double **> &x, Kokkos::View<double **> &dx) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    Kokkos::parallel_for(
        "update_weights",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t n) {
            x(i, n) += dx(i, n);
        }
    );
}