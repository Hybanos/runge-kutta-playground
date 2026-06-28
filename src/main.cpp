#include <iostream>

#include "tree.hpp"
#include "combi.hpp"

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
    int n = 5;
    int k = 4;
    std::vector<int> v(k);

    for (int i = 0; i < k; i++) {
        v[i] = i;
    }

    for (auto it = permutations(v); !it.done(); ++it) {
        for (int j = 0; j < k; j++) std::cout << (*it)[j] << " ";
        std::cout << std::endl;
    }
}

int main() {
    // test();
    // test2();
    comb();
    exit(0);
    int order = 5;
    pool p;
    p.gen(order);

    int acc = 0;
    for (int i = 0; i < order + 1; i++) {
        std::cout << "ORDER " << i << std::endl;
        for (int j = 0; j < A000081(i); j+=1) {
            if (!(j%1)) std::cout << acc <<  " " << p.to_string(acc) << std::endl;
            p.label_tree(acc);
            p.print(acc);
            acc += i;
        }
    }

    return 0;
}