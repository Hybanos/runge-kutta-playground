#pragma once

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "tree.hpp"
#include "combi.hpp"

template<class space, class layout>
struct equation_block {
    using memory_space = typename space::memory_space;
    Kokkos::View<uint8_t **, memory_space, layout> params;
    Kokkos::View<uint32_t *, memory_space, layout> sizes;
    Kokkos::View<uint32_t *, memory_space, layout> indexes;
    Kokkos::View<double *, memory_space, layout> facts;
};

template<class space, class layout>
struct jacobian_block {
    using memory_space = typename space::memory_space;
    Kokkos::View<uint8_t **, memory_space, layout> params;
    Kokkos::View<uint32_t *, memory_space, layout> sizes;
    Kokkos::View<uint32_t *, memory_space, layout> indexes;
};

using host_equations = equation_block<Kokkos::HostSpace, Kokkos::LayoutRight>;
using device_equations = equation_block<Kokkos::DefaultExecutionSpace, Kokkos::LayoutRight>;

using host_jacobian = jacobian_block<Kokkos::HostSpace, Kokkos::LayoutRight>;
using device_jacobian = jacobian_block<Kokkos::DefaultExecutionSpace, Kokkos::LayoutRight>;

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
    b_0   b_1   b_2   b_3   c_1   c_2   c_3   a_21   a_31   a_32      1      0
      0     1     2     3     4     5     6      7      8      9     10     11
*/
__inline__ uint8_t get_index(factor &f, uint8_t stages) {
    if (f.type == 'b') 
        return f.label_1;

    if (f.type == 'c') 
        return stages + 
            f.label_1 - 1;

    if (f.type == 'a') 
        return stages + stages - 1 + 
            ((f.label_1-1) * (f.label_1 - 2)) / 2 + f.label_2 - 1;
    
    return stages+1;
}

__inline__ std::string get_factor(uint8_t index, uint8_t stages) {
    uint8_t one_index = stages + stages - 1 + (stages - 1) * (stages - 2) / 2;
    uint8_t zero_index = one_index + 1;

    if (index == one_index) return "1";
    if (index == zero_index) return "0";

    if (index < stages) 
        return "b_" + std::to_string((int) index);

    if (index < stages + stages - 1)
        return "c_" + std::to_string((int) index - stages + 1);
   
    uint8_t off = index - stages - stages + 1;
    uint8_t label_1 = 0;
    uint8_t label_2 = 0;
    for (int i = 0; i < off; i++) {
        label_2++;
        if (label_2 == label_1 + 1) {
            label_1++;
            label_2 = 0;
        }
    }
    return "a_" + std::to_string((int) label_1 + 2) + std::to_string((int) label_2 + 1);
}

host_equations build_equations(pool &p, uint8_t stages);
host_jacobian build_jacobian(pool &p, uint8_t stages, host_equations &equations);

void print_equations(uint8_t stages, host_equations &equations);
void print_jacobian(uint8_t stages, host_jacobian &jacobian);