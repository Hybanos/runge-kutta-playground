#include "solve.hpp"

void init_x(Kokkos::View<double **> &x) {

    Kokkos::Random_XorShift64_Pool<> random_pool(time(NULL));

    Kokkos::parallel_for(
        "init",
        x.extent(0),
        KOKKOS_LAMBDA (uint64_t n) {
            auto generator = random_pool.get_state();
            for (int i = 0; i < x.extent(1); i++) {
                x(n, i) = generator.drand(0.0, 1.0);
            }
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
    uint64_t end[2] = {N, equations_h.total};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);
    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "equation_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t n, uint64_t i) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = equations_d.params(i, j);
                // multiply by one 1 if we're out of the param range
                if (index != total_params) prod *= x(n, index);
            }
            red(n, i) = prod;
        }
    );

    Kokkos::fence();

    // TODO: this is cursed
    for (uint64_t eq = 0; eq < equations_h.sizes.size(); eq++) {

        Kokkos::parallel_for(
            "equation_prod_reduce",
            N,
            KOKKOS_LAMBDA (uint64_t n) {
                double sum = 0;
                for (uint64_t i = 0; i < equations_d.sizes[eq]; i++) {
                    sum += red(n, equations_d.indexes[eq] + i);
                }
                f(n, eq) = sum - equations_d.facts[eq];
            }
        );
    }

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
    uint64_t end[2] = {N,jacobian_h.total};
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(begin, end);
    uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;

    Kokkos::parallel_for(
        "jacobian_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t n, uint64_t i) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = jacobian_d.params(i, j);
                // multiply by one 1 if we're out of the param range
                if (index != total_params) prod *= x(n, index);
            }
            red(n, i) = prod;
        }
    );

    Kokkos::fence();
    // simple_copy_and_print_2d(red);
    // Kokkos::fence();

    // TODO: this is cursed
    for (uint8_t derived = 0; derived < total_params; derived++) {
        for (uint64_t eq = 0; eq < (jacobian_h.sizes.size() / total_params); eq++) {
            // if (jacobian_h.sizes[eq] == 0) continue;

            uint64_t ind = derived * (jacobian_h.sizes.size() / total_params) + eq;

            Kokkos::parallel_for(
                "jacobian_prod_reduce",
                N,
                KOKKOS_LAMBDA (uint64_t n) {
                    double sum = 0;
                    for (uint64_t i = 0; i < jacobian_d.sizes[ind]; i++) {
                        sum += red(n, jacobian_d.indexes[ind] + i);
                    }
                    J(n, derived, eq) = sum;
                }
            );
        }
    }

    Kokkos::fence();
}

void transpose(Kokkos::View<double ***> &v, Kokkos::View<double ***> &vT) {
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2)});
    Kokkos::parallel_for(
        "transpose",
        policy,
        KOKKOS_LAMBDA (uint64_t n, uint64_t i, uint64_t j) {
            vT(n, j, i) = v(n, i, j);
        }
    );
    Kokkos::fence();
}

void update_weights(int N, Kokkos::View<double **> &x, Kokkos::View<double **> &dx) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    Kokkos::parallel_for(
        "update_weights",
        policy,
        KOKKOS_LAMBDA (uint64_t n, uint64_t i) {
            x(n, i) += dx(n, i);
        }
    );
}