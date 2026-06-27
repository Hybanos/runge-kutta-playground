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

pool::pool(uint64_t size) : pool_size{size} {
    _pool.resize(pool_size);
    for (uint64_t i = 0; i < pool_size; i++) _pool[i] = NODE_INIT;
}

void pool::gen(uint32_t max_order) {
    indices.resize(max_order+2);
    indices[0] = 0;
    indices[1] = 0;
    for (int32_t i = 1; i < max_order + 1; i++) {
        uint64_t tree_count = A000081(i);
        pool_size += tree_count * i;
        indices[i+1] = indices[i] + tree_count * i;
    }
    // pool_size *= 2;
    _pool.resize(pool_size);
    for (uint64_t i = 0; i < pool_size; i++) _pool[i] = NODE_INIT;
    _gen(max_order);
}

void pool::_gen(uint32_t n) {
    if (n == 1) {node_top++; return;}
    _gen(n-1);

    // loop over trees of order - 1
    for (uint32_t i = indices[n-1]; i < indices[n]; i+=n-1) {
        // std::cout << "n: " << n << "  i: " << i << std::endl;
        // std::cout << to_string(pool, i) << " " << order(pool, i) << std::endl;

        // for each leaf, copy tree while adding a new leaf at the n-th node
        for (uint32_t j = 0; j < n-1; j++) {
            uint64_t t = node_top;
            // std::cout << "new tree at " << t;
            copy_tree(*this, i, t);
            node_top += n-1;
            // std::cout << " copy : " << to_string(pool, t);
            add_leaf(*this, node_top, t, j);
            node_top++;
            // std::cout << " added : " << to_string(pool, t) << " based on " << to_string(pool, i) << std::endl;
            // print(pool, i);
            // std::cout << std::endl;
            // print(pool, t);
            sort(*this, t);
            std::string hash = to_string(*this, t);
            // std::cout << hash << std::endl;
            if (hashes.count(hash)) {
                // std::cout << "hash found ! " << hash << std::endl;
                node_top -= n;
                for (uint32_t k = 0; k < n; k++) _pool[t+k] = NODE_INIT;
            } else {
                hashes.emplace(hash);
            }
        }
    }
}

void copy_tree(pool &p, uint64_t from, uint64_t to) {
    for (uint32_t i = 0; i < order(p, from); i++) {
        p[to + i] = p[from + i];
    }
}

void add_leaf(pool &p, uint64_t nt, uint64_t t, uint32_t parent) {
    uint32_t o = order(p, t);

    node &n = p[t+parent];
    if (n.child_count == 0) {
        n.first_child = nt - t - parent;
    } else {
        uint64_t added_index = t + parent + n.first_child + n.child_count;
        // std::cout << added_index << std::endl;
        for (uint64_t i = t + o + 1; i > added_index; i--) {
            p[i] = p[i - 1];
        }
        p[added_index] = NODE_INIT;
        for (uint64_t i = t; i < added_index; i++) {
            if (i + p[i].first_child >= added_index) p[i].first_child++;
        }
    }
    n.child_count++;
}

std::string to_string(pool &p, uint64_t n) {
    node t = p[n];
    if (t.child_count == 0) return ".";
    std::string out = "[";
    for (int32_t i = 0; i < t.child_count; i++) {
        out += to_string(p, n + t.first_child + i);
    }
    return out + "]";
}

uint32_t order(pool &p, uint64_t n) {
    node t = p[n];
    uint32_t out = 1;
    for (int64_t i = 0; i < t.child_count; i++) {
        out += order(p, n + t.first_child + i);
    }
    return out;
}

int64_t fact(pool &p, uint64_t n) {
    node t = p[n];
    int64_t out = order(p, n);
    for (int32_t i = out-1; i > 0; i--) {
        out *= order(p, n + i);
    }
    return out;
}

void sort(pool &p, uint64_t n, bool rec) {
    node t = p[n];

    std::string presort = to_string(p, n);

    for (int32_t i = 0; i < t.child_count; i++) {
        sort(p, n + p[n].first_child + i);
    }

    for (int32_t i = 0; i < t.child_count-1; i++) {
        for (int32_t j = i+1; j < t.child_count; j++) {
            uint64_t c1 = n + t.first_child + i;
            uint64_t c2 = n + t.first_child + j;

            // there has to be a better way than sorting on strings
            if (to_string(p, c1) < to_string(p, c2)) {
                // std::cout << c1 << " " << c2 << std::endl;
                uint64_t tmp_fc = p[c1].first_child;
                int32_t tmp_cc = p[c1].child_count;
                p[c1].first_child = j + p[c2].first_child - i;
                p[c1].child_count = p[c2].child_count;
                p[c2].first_child = i + tmp_fc - j;
                p[c2].child_count = tmp_cc;
            }
        }
    }

    // if (!rec) std::cout << "sorting " << presort << " to " << to_string(pool, n) << std::endl;
}

void label_tree(pool &p, uint64_t n) {
    uint32_t o = order(p, n);

    char label = 'i';
    for (uint32_t i = 0; i < o; i++) {
        if (p[n+i].child_count) p[n+i].label = label++;
    }
}

void print(pool &p, uint64_t n) {
    for (uint64_t i = 0; i < order(p, n); i++) {
        std::cout << n+i << ":" << (int) p[n+i].child_count 
                << "|" << (int) p[n+i].first_child 
                << " " << p[n+i].label 
                << "  order: " << order(p, n+i)
                << "  fact: " << fact(p, n+i)
                << std::endl;
    }
}