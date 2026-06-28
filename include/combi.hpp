#pragma once

#include <vector>
#include <tuple>
#include <iterator>
#include <generator>
#include <coroutine>
#include <cstdint>

template<class T>
class permutations {
    private:
        std::vector<T> curr;
        std::vector<int> c;
        int i = 0;

        void next() {
            haha:
            if (i == curr.size()) return;
            if (c[i] < i) {
                if (!(i%2)) std::swap(curr[0], curr[i]);
                else std::swap(curr[c[i]], curr[i]);
                c[i]++;
                i = 1;
            } else {
                c[i] = 0;
                i++;
                goto haha;
            }
        }

    public:
        permutations(std::vector<T> &v) {
            curr = v;
            c.resize(v.size());
            for (int j = 0; j < v.size(); j++) {
                c[j] = 0;
            }
            i = 1;
        };

        bool done() {return i == curr.size();}
        void operator++() {next();}
        const std::vector<T> &operator*() {return curr; }
};

template<class T>
void next_k_combination(std::vector<T> &v, T n) {
    T k = v.size();
    for (T i = k - 1; i >= 0; i--) {
        if (v[i] < n - k + i + 1) {
            v[i]++;
            for (T j = i + 1; j < k; j++) {
                v[j] = v[j-1] + 1;
            }
        }
    }
}