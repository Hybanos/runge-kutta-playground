#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
// flat_set ?
#include <unordered_set>
#include <cmath>

#define MAX_TREE_ORDER 20
#define NODE_INIT {0, 0}

// https://oeis.org/A000081
uint64_t A000081(uint32_t n);

struct node {
    uint8_t first_child;
    uint8_t child_count;
    char label = 0;
};

class pool {
    private:
        std::vector<node> _pool;
        std::vector<uint64_t> indices;
        uint64_t pool_size = 0;
        uint64_t node_top = 0;
        std::unordered_set<std::string> hashes;

        void _gen(uint32_t n);

    public:
        pool(uint64_t size);
        pool() {};
        void gen(uint32_t max_order);

        uint64_t size() {return _pool.size();}
        uint64_t count_trees();

        node &operator[](uint64_t n) {return _pool[n];};

        std::string to_string(uint64_t n);
        uint32_t order(uint64_t n);
        int64_t fact(uint64_t n);
        void sort(uint64_t n, bool rec=false);

        void copy_tree(uint64_t from, uint64_t to);
        void add_leaf( uint64_t nt, uint64_t t, uint32_t parent);
        void label_tree( uint64_t n);

        void print(uint64_t n);
};

class tree_iterator {
    private: 
        pool &p;
        uint64_t n = 0;

        void next() {
            if (done()) return;
            n += p.order(n);
        }

    public:
        tree_iterator(pool &_p) : p{_p} {}
        bool done() {return n >= p.size();}
        const uint64_t operator*() {return n;}
        void operator++() {next();}
};