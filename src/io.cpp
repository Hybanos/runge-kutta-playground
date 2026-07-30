#include "io.hpp"

void wipe_checkpoint(uint8_t stages) {
    if (!std::filesystem::exists("./cache/tableaux/checkpoints")) std::filesystem::create_directories("./cache/tableaux/checkpoints/");
    std::string path = "./cache/tableaux/checkpoints/s" + std::to_string((int) stages) + ".json";
    if (std::filesystem::exists(path)) std::filesystem::remove(path);
}

void save_checkpoint(
    uint64_t N,
    uint8_t stages,
    Kokkos::View<double **> &_x, 
    Kokkos::View<double  *> &_norms, 
    Kokkos::View<double  *> &_speeds
) {

    if (!std::filesystem::exists("./cache/tableaux/checkpoints")) std::filesystem::create_directories("./cache/tableaux/checkpoints/");

    uint8_t b_offset = 0;
    uint8_t c_offset = stages;
    uint8_t a_offset = stages * 2 - 1;

    auto host_space = Kokkos::DefaultHostExecutionSpace();

    auto x      = Kokkos::create_mirror_view(host_space, _x);
    auto tmp_x  = Kokkos::create_mirror_view(host_space, _x);
    auto norms  = Kokkos::create_mirror_view_and_copy(host_space, _norms);
    auto speeds = Kokkos::create_mirror_view_and_copy(host_space, _speeds);

    Kokkos::deep_copy(tmp_x, _x);
    Kokkos::deep_copy(x, tmp_x);

    using json = nlohmann::ordered_json;

    auto j = json::array();
    for (uint64_t n = 0; n < N; n++) {
        json o;

        uint8_t b_offset = 0;
        uint8_t c_offset = stages;
        uint8_t a_offset = stages * 2 - 1;

        auto a = json::array();
        auto b = json::array();
        auto c = json::array();

        // b
        for (int i = 0; i < stages; i++) 
            b.push_back(x(b_offset + i, n));

        // c
        c.push_back(0.0);
        for (int i = 0; i < stages - 1; i++) 
            c.push_back(x(c_offset + i, n));

        // fill a with 0s
        for (int i = 0; i < stages; i++) {
            auto _a = json::array();
            for (int j = 0; j < stages; j++) {
                _a.push_back(0.0);
            }
            a.push_back(_a);
        }

        // fill a with vals
        int ind = 0;
        for (int i = 0; i < stages - 1; i++) {
            for (int j = 0; j < i; j++) {
                a[i+1][j+1] = x(a_offset + ind, n);
                ind++;
            }
        }

        // add a_x1
        for (int i = 1; i < stages; i++) {
            double sum = 0.0;
            for (int j = 0; j < stages; j++) {
                sum += a[i][j].get<double>();
            }
            a[i][0] = c[i].get<double>() - sum;
        }

        o["a"] = a;
        o["b"] = b;
        o["c"] = c;
        o["loss"] = norms(n);
        o["speed"] = speeds(n);

        j.push_back(o);
    } 

    std::ofstream out("./cache/tableaux/checkpoints/s" + std::to_string((int) stages) + ".json");
    out << std::setw(4) << j << std::endl;
    out.close();
}

bool load_checkpoint(
    uint8_t stages,
    Kokkos::View<double **> &_x, 
    Kokkos::View<double  *> &_norms, 
    Kokkos::View<double  *> &_speeds
) {
    if (!std::filesystem::exists("./cache/tableaux/checkpoints")) std::filesystem::create_directories("./cache/tableaux/checkpoints/");
    std::string path = "./cache/tableaux/checkpoints/s" + std::to_string((int) stages) + ".json";
    if (!std::filesystem::exists(path)) return false;

    using json = nlohmann::ordered_json;

    std::ifstream f;
    f.open(path);
    json o = json::parse(f);
    f.close();

    auto host_space = Kokkos::DefaultHostExecutionSpace();
    uint8_t total_params = stages * 2 - 1 + (stages - 2) * (stages - 1) / 2;
    uint64_t N = o.size();

    Kokkos::realloc(_x, total_params, N);
    Kokkos::realloc(_norms, N);
    Kokkos::realloc(_speeds, N);

    auto x      = Kokkos::create_mirror_view(host_space, _x);
    auto tmp_x  = Kokkos::create_mirror_view(host_space, x);
    auto norms  = Kokkos::create_mirror_view(host_space, _norms);
    auto speeds = Kokkos::create_mirror_view(host_space, _speeds);

    uint8_t b_offset = 0;
    uint8_t c_offset = stages;
    uint8_t a_offset = stages * 2 - 1;

    for (int n = 0; n < N; n++) {
        norms(n) = o[n]["loss"].get<double>();
        speeds(n) = o[n]["speed"].get<double>();

        for (int i = 0; i < stages; i++)
            x(b_offset + i, n) = o[n]["b"][i].get<double>();
        
        for (int i = 0; i < stages - 1; i++)
            x(c_offset + i, n) = o[n]["c"][i + 1].get<double>();

        int ind = 0;
        for (int i = 0; i < stages - 1; i++) {
            for (int j = 0; j < i; j++) {
                x(a_offset + ind, n) = o[n]["a"][i+1][j+1].get<double>();
                ind++;
            }
        }
    }

    Kokkos::deep_copy(tmp_x, x);
    Kokkos::deep_copy(_x, tmp_x);
    // Kokkos::deep_copy(_x, x);
    Kokkos::deep_copy(_norms, norms);
    Kokkos::deep_copy(_speeds, speeds);

    return true;
}

uint64_t append_solution(
    uint64_t N,
    uint8_t stages,
    Kokkos::View<double **> &_x,
    Kokkos::View<double  *> &_norms,
    double tol
) {
    if (!std::filesystem::exists("./cache/tableaux/solutions")) std::filesystem::create_directories("./cache/tableaux/solutions/");

    using json = nlohmann::ordered_json;

    uint64_t out_size;

    uint8_t b_offset = 0;
    uint8_t c_offset = stages;
    uint8_t a_offset = stages * 2 - 1;

    auto host_space = Kokkos::DefaultHostExecutionSpace();

    auto x      = Kokkos::create_mirror_view(host_space, _x);
    auto tmp_x  = Kokkos::create_mirror_view(host_space, _x);
    auto norms  = Kokkos::create_mirror_view_and_copy(host_space, _norms);

    Kokkos::fence();

    Kokkos::deep_copy(tmp_x, _x);
    Kokkos::deep_copy(x, tmp_x);

    Kokkos::fence();

    std::string path = "./cache/tableaux/solutions/s" + std::to_string((int) stages) + ".json";
    json j;
    if (std::filesystem::exists(path)) {
        std::ifstream f;
        f.open(path);
        j = json::parse(f);
        f.close();
    } else {
        j = json::array();
    }

    for (int n = 0; n < N; n++) {
        if (norms(n) > tol || Kokkos::isnan(norms(n))) continue;
        json o;

        uint8_t b_offset = 0;
        uint8_t c_offset = stages;
        uint8_t a_offset = stages * 2 - 1;

        auto a = json::array();
        auto b = json::array();
        auto c = json::array();

        // b
        for (int i = 0; i < stages; i++) 
            b.push_back(x(b_offset + i, n));

        // c
        c.push_back(0.0);
        for (int i = 0; i < stages - 1; i++) 
            c.push_back(x(c_offset + i, n));

        // fill a with 0s
        for (int i = 0; i < stages; i++) {
            auto _a = json::array();
            for (int j = 0; j < stages; j++) {
                _a.push_back(0.0);
            }
            a.push_back(_a);
        }

        // fill a with vals
        int ind = 0;
        for (int i = 0; i < stages - 1; i++) {
            for (int j = 0; j < i; j++) {
                a[i+1][j+1] = x(a_offset + ind, n);
                ind++;
            }
        }

        // add a_x1
        for (int i = 1; i < stages; i++) {
            double sum = 0.0;
            for (int j = 0; j < stages; j++) {
                sum += a[i][j].get<double>();
            }
            a[i][0] = c[i].get<double>() - sum;
        }

        o["a"] = a;
        o["b"] = b;
        o["c"] = c;
        o["loss"] = norms(n);

        j.push_back(o);
    } 

    std::ofstream out(path);
    out << std::setw(2) << j << std::endl;
    out.close();

    return j.size();
}