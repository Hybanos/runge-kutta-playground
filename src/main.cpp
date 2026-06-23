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

    pool[off + 0].first_child = 1;
    pool[off + 0].child_count = 3;
    pool[off + 2].first_child = 2;
    pool[off + 2].child_count = 1;
    pool[off + 3].first_child = 2;
    pool[off + 3].child_count = 2;

    std::cout << to_string(pool, off) << std::endl;
    sort(pool, off);
    std::cout << to_string(pool, off) << std::endl;
    std::cout << order(pool, off) << std::endl;
    std::cout << fact(pool, off) << std::endl;
}

int main() {
    // test();
    // exit(0);
    int order = 4;
    tree_manager tm(order);

    int acc = 0;
    for (int i = 0; i < order + 1; i++) {
        std::cout << "ORDER " << i << std::endl;
        for (int j = 0; j < A000081(i); j++) {
            std::cout << to_string(tm.pool, acc) << std::endl;
            print(tm.pool, acc);
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