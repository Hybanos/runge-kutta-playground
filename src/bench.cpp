#include <iostream>
// #include <filesystem>
#include <cmath>
#include <algorithm>

// #include <Kokkos_Core.hpp>
// #include <KokkosBlas3_gemm.hpp>
// #include <KokkosLapack_gesv.hpp>
#include <CLI/CLI.hpp>
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

void bench_trees() {
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
}

void bench_systems() {
    int max = 13;

    for (int i = 2; i < max; i++) {

        int64_t reps = 1 << (max - i - 1);

        std::vector<double> values(reps);

        pool p;
        p.gen(i);

        for (uint64_t j = 0; j < reps; j++) {

            auto t1 = std::chrono::high_resolution_clock::now();

            auto eq = build_equations(p, i);
            auto ja = build_jacobian(p, i, eq);

            auto t2 = std::chrono::high_resolution_clock::now();

            values[j] = (t2 - t1).count() / 1e9;
        }

        std::cout << i << " : " 
                  << avg(values) << "s average, " 
                  << mean(values) << "s mean, " 
                  << std_dev(values) << " sigmas. (" 
                  << reps << " reps)" << std::endl;
    }
}

int main(int argc, char **argv) {

    Kokkos::initialize(argc, argv);
    {
        CLI::App app("bench");
        argv = app.ensure_utf8(argv);

        std::string sel = "";

        app.add_flag("--omp");
        app.add_flag("--hip");
        app.add_flag("--cuda");

        app.add_option("-m", sel, "Bench mode");

        CLI11_PARSE(app, argc, argv);

        if (sel == "tree") bench_trees();
        if (sel == "system") bench_systems();
    }
    Kokkos::finalize();

    return 0;
}