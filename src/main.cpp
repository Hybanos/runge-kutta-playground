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
        // int order = 4;
        // pool p;
        // p.gen(order);
        
        // for (auto it = tree_iterator(p); !it.done(); ++it) {
        //     std::cout << *it << " " << p.to_string(*it) << std::endl;
        //     p.label_tree(*it);
        //     phi(p, *it).print();
        // }

        build(4);
    }
    Kokkos::finalize();

    return 0;
}