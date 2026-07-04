#include "equations.hpp"

equation_block build_equations(uint8_t order) {
    pool p;
    p.gen(order);

    uint8_t zero_index = order + order - 1 + (order - 1) * (order - 2) / 2;
    uint8_t  one_index = zero_index + 1;
    uint64_t equation_count = p.count_trees();

    std::vector<uint32_t> _equation_sizes(equation_count);
    std::vector<uint32_t> _equation_indexes(equation_count);
    std::vector<uint8_t> _factors;
    std::vector<double> _factorials(equation_count);

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
        // _phi.print();

        int label_count = 0;
        for (uint64_t i = *it; i < *it + p.order(*it); i++) if (p[i].label) label_count++;
        uint64_t local_products = 0;
        k_permutations<uint8_t> &perm = perm_iterators[label_count];
        for (; !perm.done(); ++perm) {
            bool is_zero = false;
            // for (auto &f : _phi.factors) {
            for (int i = 0; i < order; i++) {
                if (i < _phi.factors.size()) {
                    factor f = _phi.factors[i];
                    if (f.type == 'a' && (*perm)[f.label_2 - 'i'] > (*perm)[f.label_1 - 'i']) is_zero = true; 
                    if (f.type == 'c' && (*perm)[f.label_1 - 'i'] == 0) is_zero = true;

                    factor f_perm = factor{f.type, (char) (*perm)[f.label_1 - 'i'], (char) (*perm)[f.label_2 - 'i']};
                    uint8_t ind = get_index(f_perm, order);

                    // std::cout << f_perm.type << "_" << (int) f_perm.label_1 << "-" << (int) f_perm.label_2 << "\t" << (int) ind << std::endl;

                    _factors.push_back(ind);
                } else {
                    _factors.push_back(one_index);
                }
            }
            if (!is_zero) local_products++;
            else _factors.resize(_factors.size() - order);
        }
        _equation_sizes[tree_i] = local_products;
        _equation_indexes[tree_i] = total_products;
        _factorials[tree_i] = p.fact(*it);
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
    //     if (!(i%order) && i > 0) std::cout << std::endl;
    //     std::cout << (int) _factors[i] << " ";
    // }

    equation_block equations = {
        .params = Kokkos::View<uint8_t **>("", _factors.size() / order, order),
        .sizes = Kokkos::View<uint32_t *>("", _equation_sizes.size()),
        .indexes = Kokkos::View<uint32_t *>("", _equation_indexes.size()),
        .facts = Kokkos::View<double *>("", _factorials.size()),
    };

    std::memcpy(equations.params.data(), _factors.data(), _factors.size() * sizeof(uint8_t));
    std::memcpy(equations.sizes.data(), _equation_sizes.data(), _equation_sizes.size() * sizeof(uint32_t));
    std::memcpy(equations.indexes.data(), _equation_indexes.data(), _equation_indexes.size() * sizeof(uint32_t));
    std::memcpy(equations.facts.data(), _factorials.data(), _factorials.size() * sizeof(double));

    return equations;
}