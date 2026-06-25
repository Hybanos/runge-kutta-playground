#include <iostream>

#include "tree.hpp"

void test() {
    std::vector<node> pool;
    pool.resize(MAX_TREE_ORDER);

    for (int i = 0; i < MAX_TREE_ORDER; i++) {
        pool[i].first_child = -1;
        pool[i].child_count = 0;
    }

    uint64_t off = 5;

    // pool[off + 0].first_child = 1;
    // pool[off + 0].child_count = 3;
    // pool[off + 2].first_child = 2;
    // pool[off + 2].child_count = 1;
    // pool[off + 3].first_child = 2;
    // pool[off + 3].child_count = 2;

    // std::cout << to_string(pool, off) << std::endl;
    // sort(pool, off);
    // std::cout << to_string(pool, off) << std::endl;
    // std::cout << order(pool, off) << std::endl;
    // std::cout << fact(pool, off) << std::endl;

    pool[off + 0].first_child = 1;
    pool[off + 0].child_count = 1;
    pool[off + 1].first_child = 1;
    pool[off + 1].child_count = 2;
    pool[off + 2].first_child = -1;
    pool[off + 2].child_count = 0;
    pool[off + 3].first_child = 1;
    pool[off + 3].child_count = 1;
    pool[off + 4].first_child = -1;
    pool[off + 4].child_count = 0;

    std::cout << to_string(pool, off) << std::endl;
    print(pool, off);
    sort(pool, off);
    std::cout << to_string(pool, off) << std::endl;
    print(pool, off);
    std::cout << order(pool, off) << std::endl;
    std::cout << fact(pool, off) << std::endl;
}

void test2() {
    std::vector<node> pool;
    pool.resize(MAX_TREE_ORDER * 10);

    for (int i = 0; i < MAX_TREE_ORDER * 10; i++) {
        pool[i].first_child = -1;
        pool[i].child_count = 0;
    }

    pool[0].first_child = 1;
    pool[0].child_count = 1;
    pool[1].first_child = 1;
    pool[1].child_count = 1;
    pool[2].first_child = 1;
    pool[2].child_count = 1;

    uint64_t node_top = 4;
    copy_tree(pool, 0, node_top);
    node_top += 4;
    add_leaf(pool, node_top, 4, 1);
    std::cout << to_string(pool, 0) << std::endl;
    print(pool, 0);

    std::cout << to_string(pool, 4) << std::endl;
    print(pool, 4);
}

int main() {
    // test();
    // test2();
    // exit(0);
    int order = 15;
    tree_manager tm(order);

    int acc = 0;
    for (int i = 0; i < order + 1; i++) {
        std::cout << "ORDER " << i << std::endl;
        for (int j = 0; j < A000081(i); j+=1) {
            if (!j%100) std::cout << acc <<  " " << to_string(tm.pool, acc) << std::endl;
            // print(tm.pool, acc);
            acc += i;
        }
    }
    // std::cout << to_string(tm.pool, 0) << std::endl;
    // std::cout << to_string(tm.pool, 1) << std::endl;
    // std::cout << to_string(tm.pool, 3) << std::endl;
    // std::cout << to_string(tm.pool, 6) << std::endl;

    // print(tm.pool, 0);
    // print(tm.pool, 1);
    // print(tm.pool, 3);
    // print(tm.pool, 6);

    return 0;
}