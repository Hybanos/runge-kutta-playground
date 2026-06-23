#include "tree.hpp"

uint64_t A000081(uint32_t n) {
    if (n <= 1) return n;

    int s = 0;
    for (int k = 1; k < n; k++) {
        int ss = 0;
        for (int d = 1; d < k+1; d++) {
            if (!(k % d)) ss += d * A000081(d);
        }
        s = s + ss * A000081(n-k);
    }
    return s / (n-1);
}

tree_manager::tree_manager(uint32_t max_order) {
    indices.resize(max_order+2);
    indices[0] = 0;
    indices[1] = 0;
    for (int32_t i = 1; i < max_order + 1; i++) {
        uint64_t tree_count = A000081(i);
        pool_size += tree_count * i;
        indices[i+1] = indices[i] + tree_count;
    }
    pool.resize(pool_size);
    for (uint64_t i = 0; i < pool_size; i++) {
        pool[i].first_child = -1;
        pool[i].child_count = 0;
    }
    std::cout << indices[0] << std::endl;
    std::cout << indices[1] << std::endl;
    std::cout << indices[2] << std::endl;
    std::cout << indices[3] << std::endl;
    std::cout << indices[4] << std::endl;
    // std::cout << indices[5] << std::endl;
    gen(max_order);
}

void tree_manager::gen(uint32_t n) {
    if (n == 1) {node_top++; return;}
    gen(n-1);

    // loop over trees of order - 1
    for (uint32_t i = indices[n-1]; i < indices[n]; i++) {
        std::cout << "n: " << n << "  i: " << i << std::endl;
        // std::cout << to_string(pool, i) << " " << order(pool, i) << std::endl;
        // for each leaf, copy tree while adding a new leaf at the n-th node
        for (uint32_t j = 0; j < n-1; j++) {
            uint64_t t = node_top;
            copy_tree(i, t);
            add_leaf(t, j);
            std::string hash = to_string(pool, j);
            if (hashes.count(hash)) {
                std::cout << "hash found ! " << hash << std::endl;
                node_top -= n;
                for (uint32_t k = 0; k < n; k++) pool[t+k] = {-1, -1};
            } else {
                hashes.emplace(hash);
            }
        }
    }
}

void tree_manager::copy_tree(uint64_t from, uint64_t to) {
    for (uint32_t i = 0; i < order(pool, from); i++) {
        pool[to + i] = pool[from + i];
        node_top++;
    }
}

// absolutely does fuck up the next tree in the pool
void tree_manager::add_leaf(uint64_t t, uint32_t parent) {
    uint32_t o = order(pool, t);

    node n = pool[t+parent];
    if (n.child_count == 0) {
        n.first_child = node_top;
    } else {
        uint64_t to_shift = n.first_child + n.child_count;
        memcpy(&pool[to_shift+1], &pool[to_shift], MAX_TREE_ORDER * sizeof(node));
        for (uint64_t i = t+parent; i < to_shift + MAX_TREE_ORDER; i++) {
            if (pool[i].first_child <= -1) {
                pool[i].first_child += 1;
            }
        }
    }
    n.child_count++;
    node_top++;
}

std::string to_string(std::vector<node> &pool, uint64_t n) {
    node t = pool[n];
    if (t.child_count == 0) return ".";
    std::string out = "[";
    for (int32_t i = 0; i < t.child_count; i++) {
        out += to_string(pool, n + t.first_child + i);
    }
    return out + "]";
}

uint32_t order(std::vector<node> &pool, uint64_t n) {
    node t = pool[n];
    uint32_t out = 1;
    for (int64_t i = 0; i < t.child_count; i++) {
        out += order(pool, n + t.first_child + i);
    }
    return out;
}

int64_t fact(std::vector<node> &pool, uint64_t n) {
    node t = pool[n];
    int64_t out = order(pool, n);
    for (int32_t i = 0; i < t.child_count; i++) {
        out *= order(pool, n + t.first_child + i);
    }
    return out;
}

void sort(std::vector<node> &pool, uint64_t n) {
    node t = pool[n];

    for (int32_t i = 0; i < t.child_count-1; i++) {
        for (int32_t j = i+1; j < t.child_count; j++) {
            uint64_t c1 = n + t.first_child + i;
            uint64_t c2 = n + t.first_child + j;

            if (order(pool, c1) < order(pool, c2)) {
                std::swap(pool[c1], pool[c2]);
            }
        }
    }
}

void print(std::vector<node> &pool, uint64_t n) {
    node t = pool[n];

    std::cout << n << ":" << t.child_count << "|" << t.first_child << std::endl;
    for (int32_t i = 0; i < t.child_count; i++) {
        print(pool, n + t.first_child + i);
    }
}