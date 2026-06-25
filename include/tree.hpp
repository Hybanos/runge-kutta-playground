#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <unordered_set>

#define MAX_TREE_ORDER 20

struct node {
    int64_t first_child;
    int32_t child_count;
};

class tree_manager {
    private:
        std::vector<uint64_t> indices;
        std::vector<uint64_t> nodes_accumulated;
        uint64_t pool_size = 0;
        uint64_t node_top = 0;
        std::unordered_set<std::string> hashes;

        uint64_t get_start(uint32_t order);
        void gen(uint32_t n);
    public:
        std::vector<node> pool;

        tree_manager(uint32_t max_order);
};

// https://oeis.org/A000081
uint64_t A000081(uint32_t n);

std::string to_string(std::vector<node> &pool, uint64_t n);
uint32_t order(std::vector<node> &pool, uint64_t n);
int64_t fact(std::vector<node> &pool, uint64_t n);
void sort(std::vector<node> &pool, uint64_t n, bool rec=false);

void copy_tree(std::vector<node> &pool, uint64_t from, uint64_t to);
void add_leaf(std::vector<node> &pool, uint64_t nt, uint64_t t, uint32_t parent);

void print(std::vector<node> &pool, uint64_t n);
