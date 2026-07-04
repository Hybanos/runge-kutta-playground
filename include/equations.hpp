#pragma once

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "tree.hpp"

template<class execspace, class layout>
struct equation_block {
    // Kokkos::View<uint8_t **, execspace> params;
    // Kokkos::View<uint32_t *, execspace> sizes;
    // Kokkos::View<double *, layout, execspace> facts;
};

using CPU_equations = equation_block<Kokkos::HostSpace, Kokkos::layout_right>;
using GPU_equations = equation_block<Kokkos::DefaultExecutionSpace, Kokkos::layout_left>;

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

CPU_equations build(uint8_t order) {
    pool p;
    p.gen(order);

    std::vector<uint32_t> _sizes(p.count_trees());
    std::vector<uint8_t> _factors;
    std::vector<double> _factorials(p.count_trees());

    std::vector<uint8_t> label_values(order); 
    for (int i = 0; i < order; i++) label_values[i] = i;

    // cache all k-permutation we're gonna need (x25 banger speedup)
    std::vector<k_permutations<uint8_t>> perm_iterators;
    for (int i = 0; i < order; i++) {perm_iterators.push_back(k_permutations(i, label_values));std::cout << "alloc done " << i << std::endl;}

    uint64_t total_products = 0;
    uint64_t tree_i = 0;
    for (auto it = tree_iterator(p); !it.done(); ++it) {
        p.label_tree(*it);
        phi _phi(p, *it);
        _phi.print();

        int label_count = 0;
        for (uint64_t i = *it; i < *it + p.order(*it); i++) if (p[i].label) label_count++;
        uint64_t local_products = 0;
        k_permutations<uint8_t> &perm = perm_iterators[label_count];
        for (; !perm.done(); ++perm) {
            bool is_zero = false;
            for (auto &f : _phi.factors) {
                if (f.type == 'a' && (*perm)[f.label_2 - 'i'] > (*perm)[f.label_1 - 'i']) is_zero = true; 
                if (f.type == 'c' && (*perm)[f.label_1 - 'i'] == 0) is_zero = true;

                factor f_perm = factor{f.type, (char) (*perm)[f.label_1 - 'i'], (char) (*perm)[f.label_2 - 'i']};
                uint8_t ind = get_index(f_perm, order);

                std::cout << f_perm.type << "_" << (int) f_perm.label_1 << "-" << (int) f_perm.label_2 << "\t" << (int) ind << std::endl;

                _factors.push_back(ind);
            }
            if (!is_zero) local_products++;
            else _factors.resize(_factors.size() - order);
        }
        _sizes[tree_i] = local_products;
        _factorials[tree_i] = p.fact(*it);
        total_products += local_products;
        tree_i++;
    }

    std::cout << total_products << std::endl;
    for (int i = 0; i < _sizes.size(); i++) {
        std::cout << _sizes[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < _factors.size(); i++) {
        std::cout << (int) _factors[i] << " ";
        if (!(i%order) && i > 0) std::cout << std::endl;
    }

    return CPU_equations{};
}