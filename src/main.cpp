#include <iostream>

#include "tree.hpp"

void test() {
    std::vector<node> pool;
    pool.resize(MAX_TREE_ORDER);

    for (int i = 0; i < MAX_TREE_ORDER; i++) {
        pool[i].first_child = -1;
        pool[i].child_count = 0;
    }

    pool[0].first_child = 1;
    pool[0].child_count = 2;
    pool[2].first_child = 1;
    pool[2].child_count = 1;

    std::cout << to_string(pool, 0) << std::endl;
    sort(pool, 0);
    std::cout << to_string(pool, 0) << std::endl;
    std::cout << order(pool, 0) << std::endl;
    std::cout << fact(pool, 0) << std::endl;
}

int main() {
    // test();
    // exit(0);
    tree_manager tm(2);
    // for (int i = 0; i < tm.pool.size(); i++) {
    //     std::cout << to_string(tm.pool, i) << std::endl;
    //     // print(tm.pool, i);
    // }
    std::cout << to_string(tm.pool, 0) << std::endl;
    std::cout << to_string(tm.pool, 1) << std::endl;
    // std::cout << to_string(tm.pool, 3) << std::endl;
    // std::cout << to_string(tm.pool, 6) << std::endl;

    print(tm.pool, 0);
    print(tm.pool, 1);
    // print(tm.pool, 3);
    // print(tm.pool, 6);

    return 0;
}