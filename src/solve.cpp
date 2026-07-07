#include "solve.hpp"

void init_x(Kokkos::View<double **> &x) {
    Kokkos::parallel_for(
        "init",
        x.extent(0),
        KOKKOS_LAMBDA (uint64_t n) {
            for (int i = 0; i < x.extent(1) - 2; i++) {
                x(n, i) = 0.5;
            }
            x(n, x.extent(1) - 2) = 1;
            x(n, x.extent(1) - 1) = 0;
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

    Kokkos::parallel_for(
        "equation_prod_evaluate", 
        policy, 
        KOKKOS_LAMBDA (uint64_t n, uint64_t i) {
            double prod = 1.0;
            for (int j = 0; j < stages; j++) {
                uint8_t index = equations_d.params(i, j);
                prod *= x(n, index);
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
                f(n, eq) = sum;
            }
        );
    }

    Kokkos::fence();
}

void simple_copy_and_print_2d(Kokkos::View<double **> &v) {
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