#include <iostream>

#include "tree.hpp"

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

    std::cout << to_string(p, off) << std::endl;
    print(p, off);
    sort(p, off);
    std::cout << to_string(p, off) << std::endl;
    print(p, off);
    std::cout << order(p, off) << std::endl;
    std::cout << fact(p, off) << std::endl;
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
    copy_tree(p, 0, node_top);
    node_top += 4;
    add_leaf(p, node_top, 4, 1);
    std::cout << to_string(p, 0) << std::endl;
    print(p, 0);

    std::cout << to_string(p, 4) << std::endl;
    print(p, 4);
}

int main() {
    // test();
    // test2();
    // exit(0);
    int order = 6;
    pool p;
    p.gen(order);

    int acc = 0;
    for (int i = 0; i < order + 1; i++) {
        std::cout << "ORDER " << i << std::endl;
        for (int j = 0; j < A000081(i); j+=1) {
            if (!(j%1)) std::cout << acc <<  " " << to_string(p, acc) << std::endl;
            print(p, acc);
            acc += i;
        }
    }

    return 0;
}