#pragma once

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "tree.hpp"
#include "combi.hpp"

struct equation_block {
    Kokkos::View<uint8_t **>params;
    Kokkos::View<uint32_t *> sizes;
    Kokkos::View<uint32_t *> indexes;
    Kokkos::View<double *> facts;
};

struct factor {
    char type = 0;
    char label_1 = 0;
    char label_2 = 0;
};

struct phi {
    std::vector<factor> factors;

    void print() {
        for (int i = 0; i < factors.size(); i++) {
            factor f = factors[i];
            std::cout << f.type << "_" << f.label_1;
            if (f.label_2) std::cout << f.label_2;
            if (i < factors.size() - 1) std::cout << "*";
        }
        std::cout << std::endl;
    }

    phi(pool p, uint64_t n) {
        int o = p.order(n);
        factors.reserve(o);
        factors.push_back(factor{'b', p[n].label, 0});
        for (int i = n; i < n + o; i++) {
            for (int j = i+p[i].first_child; j < i + p[i].first_child + p[i].child_count; j++) {
                if (p[j].child_count) factors.push_back(factor{'a', p[i].label, p[j].label});
                else factors.push_back(factor{'c', p[i].label, 0});
            }
        } 
    }
};

// maps the butcher tableau factor to its slot in the linear array
/*
    for order 4: 
    b_0   b_1   b_2   b_3   c_1   c_2   c_3   a_21   a_31   a_32      0      1
      0     1     2     3     4     5     6      7      8      9     10     11
*/
__inline__ uint8_t get_index(factor &f, uint8_t order) {
    if (f.type == 'b') 
        return f.label_1;

    if (f.type == 'c') 
        return order + 
            f.label_1 - 1;

    if (f.type == 'a') 
        return order + order - 1 + 
            ((f.label_1-1) * (f.label_1 - 2)) / 2 + f.label_2 - 1;
}

equation_block build_equations(uint8_t order);
equation_block build_jacobian(equation_block equations);