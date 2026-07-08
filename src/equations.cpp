#include "equations.hpp"

host_equations build_equations(pool &p, uint8_t stages) {

    uint8_t one_index = stages + stages - 1 + (stages - 1) * (stages - 2) / 2;
    uint64_t equation_count = p.count_trees();

    std::vector<uint8_t> _factors;
    std::vector<uint32_t> _equation_sizes(equation_count);
    std::vector<uint32_t> _equation_indexes(equation_count);
    std::vector<double> _factorials(equation_count);

    std::vector<uint8_t> label_values(stages); 
    for (int i = 0; i < stages; i++) label_values[i] = i;

    // cache all k-permutation we're gonna need (x25 banger speedup)
    std::vector<k_permutations<uint8_t>> perm_iterators;
    for (int i = 0; i < stages; i++) {perm_iterators.push_back(k_permutations(i, label_values));std::cout << "alloc done " << i << std::endl;}

    uint64_t total_products = 0;
    uint64_t tree_i = 0;
    for (auto it = tree_iterator(p); !it.done(); ++it) {
        p.label_tree(*it);
        phi _phi(p, *it);
        // _phi.print();

        int label_count = 0;
        for (uint64_t i = *it; i < *it + p.order(*it); i++) if (p[i].label) label_count++;
        uint64_t local_products = 0;
        k_permutations<uint8_t> &perm = perm_iterators[label_count];
        for (; !perm.done(); ++perm) {
            bool is_zero = false;
            // for (auto &f : _phi.factors) {
            for (int i = 0; i < stages; i++) {
                if (i < _phi.factors.size()) {
                    factor f = _phi.factors[i];
                    if (f.type == 'a' && (*perm)[f.label_2 - 'i'] > (*perm)[f.label_1 - 'i']) is_zero = true; 
                    if (f.type == 'c' && (*perm)[f.label_1 - 'i'] == 0) is_zero = true;

                    factor f_perm = factor{f.type, (char) (*perm)[f.label_1 - 'i'], (char) (*perm)[f.label_2 - 'i']};
                    uint8_t ind = get_index(f_perm, stages);

                    // std::cout << f_perm.type << "_" << (int) f_perm.label_1 << "-" << (int) f_perm.label_2 << "\t" << (int) ind << std::endl;

                    _factors.push_back(ind);
                } else {
                    _factors.push_back(one_index);
                }
            }
            if (!is_zero) local_products++;
            else _factors.resize(_factors.size() - stages);
        }
        _equation_sizes[tree_i] = local_products;
        _equation_indexes[tree_i] = total_products;
        _factorials[tree_i] = 1.0 / p.fact(*it);
        total_products += local_products;
        tree_i++;
    }

    // std::cout << total_products << std::endl;
    // for (int i = 0; i < _equation_sizes.size(); i++) {
    //     std::cout << _equation_sizes[i] << "\t";
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < _equation_sizes.size(); i++) {
    //     std::cout << _equation_indexes[i] << "\t";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < _factors.size(); i++) {
    //     if (!(i%stages) && i > 0) std::cout << std::endl;
    //     std::cout << (int) _factors[i] << " ";
    // }

    host_equations equations {
        decltype(host_equations::params)("", _factors.size() / stages, stages),
        decltype(host_equations::sizes)("", _equation_sizes.size()),
        decltype(host_equations::indexes)("", _equation_indexes.size()),
        decltype(host_equations::facts)("", _factorials.size()),
        _equation_indexes[equation_count-1] + _equation_sizes[equation_count-1] 
    };
    
    // Kokkos::View<uint32_t *, Kokkos::HostSpace, Kokkos::LayoutRight> equation_sizes("", _equation_sizes.size());
    // host_equations equations {
    //     decltype(host_equations::params)("", _factors.size() / stages, stages),
    //     equation_sizes
    // }

    std::memcpy(equations.params.data(), _factors.data(), _factors.size() * sizeof(uint8_t));
    std::memcpy(equations.sizes.data(), _equation_sizes.data(), _equation_sizes.size() * sizeof(uint32_t));
    std::memcpy(equations.indexes.data(), _equation_indexes.data(), _equation_indexes.size() * sizeof(uint32_t));
    std::memcpy(equations.facts.data(), _factorials.data(), _factorials.size() * sizeof(double));

    return equations;
}

host_jacobian build_jacobian(pool &p, uint8_t stages, host_equations &equations) {

    uint8_t param_count = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;
    uint8_t one_index = param_count;
    uint8_t zero_index = one_index + 1;
    uint64_t equation_count = p.count_trees() * param_count;

    std::vector<uint8_t> _factors;
    std::vector<uint32_t> _equation_sizes(equation_count);
    std::vector<uint32_t> _equation_indexes(equation_count);
   
    uint64_t total_products = 0;
    uint64_t equation_i = 0;
    for (int j = 0; j < param_count; j++) {
        for (int i = 0; i < equations.sizes.size(); i++) {
            uint32_t size = equations.sizes[i];
            uint32_t index = equations.indexes[i];
            uint64_t local_products = 0;
            // derivate equation 
            for (int k = 0; k < size; k += 1) {
                bool factor_found = false;
                // derivate product
                for (int l = 0; l < stages; l++) {
                    uint8_t prod = equations.params(index + k, l);
                    if (prod == j && !factor_found) {
                        _factors.push_back(one_index);
                        factor_found = true;
                    } else {
                        _factors.push_back(prod); 
                    }
                }
                
                // if we didn't derivate yet, set product to 0
                if (!factor_found) {
                    _factors.resize(_factors.size() - stages);
                } else {
                    local_products++;
                }
            }
            _equation_sizes[equation_i] = local_products;
            _equation_indexes[equation_i] = total_products;
            total_products += local_products;
            equation_i++;
        }
    }

    host_jacobian jacobian {
        decltype(host_jacobian::params)("", _factors.size() / stages, stages),
        decltype(host_jacobian::sizes)("", _equation_sizes.size()),
        decltype(host_jacobian::indexes)("", _equation_indexes.size()),
        _equation_indexes[equation_count-1] + _equation_sizes[equation_count-1] 
    };

    std::memcpy(jacobian.params.data(), _factors.data(), _factors.size() * sizeof(uint8_t));
    std::memcpy(jacobian.sizes.data(), _equation_sizes.data(), _equation_sizes.size() * sizeof(uint32_t));
    std::memcpy(jacobian.indexes.data(), _equation_indexes.data(), _equation_indexes.size() * sizeof(uint32_t));

    return jacobian;
}

void print_equations(uint8_t stages, host_equations &equations) {
    std::cout << "=== Equations ===" << std::endl;
    for (int i = 0; i < equations.sizes.size(); i++) {
        uint32_t size = equations.sizes[i];
        uint32_t index = equations.indexes[i];

        for (int k = 0; k < size; k += 1) {
            for (int l = 0; l < stages; l++) {
                uint8_t prod = equations.params(index + k, l);

                std::cout << get_factor(prod, stages);
                if (l < stages - 1) std::cout << "*";
            }
            if (k < size - 1) std::cout << " + ";
            else std::cout << " - 1/" << equations.facts[i];
        }
        std::cout << std::endl;
    }    
}

void print_jacobian(uint8_t stages, host_jacobian &jacobian) {
    uint8_t param_count = (stages - 1) / (stages - 2) / 2 + stages + stages - 1;
    int derivation_param = -1;

    std::cout << "=== Jacobian ===" << std::endl;
    for (int i = 0; i < jacobian.sizes.size(); i++) {
        uint32_t size = jacobian.sizes[i];
        uint32_t index = jacobian.indexes[i];

        if (i % (jacobian.sizes.size() / param_count) == 0) {
            derivation_param++;
            std::cout << "derivating over " + get_factor(derivation_param, stages) << ":" << std::endl;
        }

        for (int k = 0; k < size; k += 1) {
            for (int l = 0; l < stages; l++) {
                uint8_t prod = jacobian.params(index + k, l);

                // std::cout << get_factor(prod, stages);
                std::cout << (int) prod;
                if (l < stages - 1) std::cout << "*";
            }
            if (k < size - 1) std::cout << " + ";
        }
        if (size == 0) std::cout << "0";
        std::cout << std::endl;
    }    
}