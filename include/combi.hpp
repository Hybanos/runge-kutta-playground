#pragma once

#include <vector>
#include <tuple>
#include <iterator>
#include <coroutine>
#include <cstdint>

template<class T>
class permutations {
    private:
        std::vector<T> &curr;
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
        permutations(std::vector<T> &v) : curr{v} {
            c.resize(v.size());
            for (uint64_t j = 0; j < v.size(); j++) {
                c[j] = 0;
            }
            i = 1;
        };

        bool done() {return i == curr.size();}
        void operator++() {next();}
        const std::vector<T> &operator*() {return curr; }
};

// TODO: make good
template<class T>
class k_permutations {
    private:
        uint64_t i = 0;
        uint64_t iter = 0;
        uint64_t k;
        uint64_t total;
        std::vector<T> &v;
        std::vector<T> out;
        std::unordered_set<uint64_t> hashes;

        uint64_t checksum() {
            uint64_t c = 0;
            for (int j = 0; j < k; j++) {
                c += out[i * k + j];
                c = c << 6 | c >> (64 - 6);
                c ^= ~static_cast<uint64_t>(0);
            }
            return c;
        }

        void gen() {
            for (auto it = permutations(v); !it.done(); ++it) {

                for (int n = 0; n < k; n++) out[i * k + n] = (*it)[n];
                auto hash = checksum();
                if (!hashes.contains(hash)) {
                    hashes.emplace(hash);
                    i++;
                }

                if (i == total) break;
            }
        }

    public:
        k_permutations(int _k, std::vector<T> &_v) : k{_k}, v{_v} {
            total = tgammal(v.size() + 1) / tgammal(v.size() - k + 1);
            if (k == 1) total = v.size();
            out.resize(k * total);
            gen();
            hashes.clear();
        }

        bool done() {
            bool d = iter >= total; 
            if (d) {
                iter = 0;
            }
            return d;
        }
        void operator++() {iter++;}
        T* operator*() {return out.data() + iter * k;}
};