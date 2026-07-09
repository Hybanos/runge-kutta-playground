#include "solve.hpp"

void init_x(Kokkos::View<double **> &x) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {x.extent(0), x.extent(1)});
    Kokkos::Random_XorShift64_Pool<> random_pool(time(NULL));
    // Kokkos::Random_XorShift64_Pool<> random_pool(0);

    Kokkos::parallel_for(
        "init",
        policy,
        KOKKOS_LAMBDA (uint64_t i, uint64_t j) {
            auto generator = random_pool.get_state();
            // x(i, j) = generator.drand(0.0, 1.0);
            x(i, j) = 0.1 * (i+1);
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

    // TODO: this is cursed
    for (uint64_t eq = 0; eq < equations_h.sizes.size(); eq++) {

        Kokkos::parallel_for(
            "equation_prod_reduce",
            N,
            KOKKOS_LAMBDA (uint64_t n) {
                double sum = 0;
                for (uint64_t i = 0; i < equations_d.sizes[eq]; i++) {
                    sum += red(equations_d.indexes[eq] + i, n);
                }
                f(eq, n) = sum - equations_d.facts[eq];
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
    simple_copy_and_print_2d(red);
    Kokkos::fence();

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
                        sum += red(jacobian_d.indexes[ind] + i, n);
                    }
                    J(derived, eq, n) = sum;
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