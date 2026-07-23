#include "io.hpp"

void save_to_json(
    uint64_t N,
    uint8_t stages,
    Kokkos::View<double **> &_x, 
    Kokkos::View<double  *> &_norms, 
    Kokkos::View<double  *> &_speeds
) {

    if (!std::filesystem::exists("./cache/tableaux")) std::filesystem::create_directories("./cache/tableaux");

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

    using json = nlohmann::json;

    auto j = json::array();
    for (uint64_t n = 0; n < N; n++) {
        json o;

        auto a = json::array();
        auto b = json::array();
        auto c = json::array();

        for (int i = 0; i < stages; i++) 
            b.push_back(x(b_offset + i, n));

        c.push_back(0);
        for (int i = 0; i < stages - 1; i++) 
            c.push_back(x(c_offset + i, n));

        for (int i = 0; i < stages; i++) {
            auto _a = json::array();
            for (int j = 0; j < stages; j++) {
                if (i == 0 || j == 0) 
                    _a.push_back(0);
                else 
                    _a.push_back(x(a_offset + (j * stages - 1) + i, n));
            }
            a.push_back(_a);
        }

        o["a"] = a;
        o["b"] = b;
        o["c"] = c;
        o["loss"] = norms(n);
        o["speed"] = speeds(n);

        j.push_back(o);
    } 

    std::ofstream out("./cache/tableaux/s" + std::to_string((int) stages));
    out << std::setw(4) << j << std::endl;
}