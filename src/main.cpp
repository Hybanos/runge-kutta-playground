#include <iostream>

#include <Kokkos_Core.hpp>

#include "tree.hpp"
#include "combi.hpp"
#include "equations.hpp"

void test() {
    pool p(MAX_TREE_ORDER);

    for (int i = 0; i < MAX_TREE_ORDER; i++) {
        p[i].first_child = -1;
        p[i].child_count = 0;
    }

    uint64_t off = 5;

    p[off + 0].first_child = 1;
    p[off + 0].child_count = 1;
    p[off + 1].first_child = 1;
    p[off + 1].child_count = 2;
    p[off + 2].first_child = -1;
    p[off + 2].child_count = 0;
    p[off + 3].first_child = 1;
    p[off + 3].child_count = 1;
    p[off + 4].first_child = -1;
    p[off + 4].child_count = 0;

    std::cout << p.to_string(off) << std::endl;
    p.print(off);
    p.sort(off);
    std::cout << p.to_string(off) << std::endl;
    p.print(off);
    std::cout << p.order(off) << std::endl;
    std::cout << p.fact(off) << std::endl;
}

void test2() {
    pool p(MAX_TREE_ORDER * 10);

    for (int i = 0; i < MAX_TREE_ORDER * 10; i++) {
        p[i].child_count = 0;
    }

    p[0].first_child = 1;
    p[0].child_count = 1;
    p[1].first_child = 1;
    p[1].child_count = 1;
    p[2].first_child = 1;
    p[2].child_count = 1;

    uint64_t node_top = 4;
    p.copy_tree(0, node_top);
    node_top += 4;
    p.add_leaf(node_top, 4, 1);
    std::cout << p.to_string(0) << std::endl;
    p.print(0);

    std::cout << p.to_string(4) << std::endl;
    p.print(4);
}

void comb() {
    int n = 7;
    int k = 1;
    std::vector<uint8_t> v(n);
    // std::vector<int> v(n);

    for (int i = 0; i < n; i++) {
        v[i] = i;
    }

    for (auto it = permutations(v); !it.done(); ++it) {
        for (int j = 0; j < n; j++) std::cout << (int) (*it)[j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    auto it = k_permutations(k, v);
    for (; !it.done(); ++it) {
        for (int j = 0; j < k; j++) std::cout << (int) (*it)[j] << " ";
        std::cout << std::endl;
    }
}

int main() {
    Kokkos::initialize();
    {
        // test();
        // test2();
        // comb();
        // exit(0);

        // generate trees
        uint8_t stages = 4;
        pool p;
        p.gen(stages);

        auto device_space = Kokkos::DefaultExecutionSpace();

        // build equation array and jacobian matrix 
        host_equations equations_h = build_equations(p, stages);
        host_jacobian jacobian_h = build_jacobian(p, stages, equations_h);

        print_equations(stages, equations_h);
        print_jacobian(stages, jacobian_h);

        // auto tmp = Kokkos::create_mirror_view(equations_h.params);

        // copy
        device_equations equations_d = {
            .params = decltype(device_equations::params)("", equations_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.sizes),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, equations_h.indexes),
            .facts = Kokkos::create_mirror_view_and_copy(device_space, equations_h.facts),
        };

        device_jacobian jacobian_d = {
            .params = decltype(device_jacobian::params)("", jacobian_h.params.extents()),
            .sizes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.sizes),
            .indexes = Kokkos::create_mirror_view_and_copy(device_space, jacobian_h.indexes),
        };

        auto tmp_equation_alloc = Kokkos::create_mirror_view(equations_d.params);
        Kokkos::deep_copy(tmp_equation_alloc, equations_h.params);
        Kokkos::deep_copy(equations_d.params, tmp_equation_alloc);
        
        auto tmp_jacobian_alloc = Kokkos::create_mirror_view(jacobian_d.params);
        Kokkos::deep_copy(tmp_jacobian_alloc, jacobian_h.params);
        Kokkos::deep_copy(jacobian_d.params, tmp_jacobian_alloc);

        // while(true) {}

        // while true:
            // solve system
    }
    Kokkos::finalize();

    return 0;
}