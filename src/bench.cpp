#include <iostream>
// #include <filesystem>
#include <cmath>
#include <algorithm>

// #include <Kokkos_Core.hpp>
// #include <KokkosBlas3_gemm.hpp>
// #include <KokkosLapack_gesv.hpp>
// #include <CLI/CLI.hpp>
// #include <thread>

#include "tree.hpp"
#include "combi.hpp"
#include "equations.hpp"
#include "solve.hpp"
#include "io.hpp"

double mean(std::vector<double> &v) {
    std::sort(v.begin(), v.end());

    int half = v.size() / 2;

    return v[half];
}

double avg(std::vector<double> &v) {
    double acc = 0;

    for (int i = 0; i < v.size(); i++) {
        acc += v[i];
    }

    return acc / v.size();
}

double std_dev(std::vector<double> &v) {
    double average = avg(v);

    double variance = 0;

    for (int i = 0; i < v.size(); i++) {
        variance += (v[i] - average) * (v[i] - average);
    }

    variance /= v.size();

    return std::sqrt(variance);
}

int main(int argc, char **argv) {

    for (int i = 2; i < 20; i++) {

        int64_t reps = i < 16 ? 1 << (16 - i) : 1;
        std::vector<double> values(reps);

        for (uint64_t j = 0; j < reps; j++) {

            //warmup
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }

            auto t1 = std::chrono::high_resolution_clock::now();

            { pool p; p.gen(i); }
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }
            { pool p; p.gen(i); }

            auto t2 = std::chrono::high_resolution_clock::now();

            values[j] = (t2 - t1).count() / 1e9 / 7.0;
        }

        std::cout << i << " : " 
                  << avg(values) << "s average, " 
                  << mean(values) << "s mean, " 
                  << std_dev(values) << " sigmas. (" 
                  << reps << " reps)" << std::endl;
    }

    return 0;
}