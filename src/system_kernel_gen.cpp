#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include "tree.hpp"
#include "combi.hpp"
#include "solve.hpp"

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {
        if (argc != 3) {
            std::cout << "usage: <exec> order stages" << std::endl;
            exit(0);
        }

        uint8_t order = atoi(argv[1]);
        uint8_t stages = atoi(argv[2]);

        pool p;
        p.gen(order);

        uint8_t total_params = (stages - 1) * (stages - 2) / 2 + stages + stages - 1;
        host_equations equations_h = build_equations_or_get_cached(p, order, stages);
        host_jacobian jacobian_h = build_jacobian_or_get_cached(p, order, stages, equations_h);

        if (!std::filesystem::exists("./include/gen/")) std::filesystem::create_directories("./include/gen/");

        std::string path = "./include/gen/p" + std::to_string((int) order) + "_s" + std::to_string((int) stages) + ".hpp";
        std::ofstream out(path);
        
        out << R"(#include <Kokkos_Core.hpp>

#include "common.hpp"

void eval_generated_equations(Kokkos::View<double **> &x, Kokkos::View<double **> &f, uint64_t n) {
)";

        for (int i = 0; i < total_params; i++) {
            out << "    double " + get_factor(i, stages) << " = x(" << std::to_string(i) << ", n);" << std::endl;
        }

        out << std::endl;

        for (int i = 0; i < equations_h.sizes.size(); i++) {
            uint32_t size = equations_h.sizes[i];
            uint32_t index = equations_h.indexes[i];

            out << "    f(" << i << ", n) = ";

            for (int k = 0; k < size; k += 1) {
                for (int l = 0; l < order; l++) {
                    uint8_t prod = equations_h.params(index + k, l);

                    out << get_factor(prod, stages);
                    if (l < order - 1) out << "*";
                }
                if (k < size - 1) out << " + ";
                else out << " - " << equations_h.facts[i];
            }
            out << ";" << std::endl;
        }    
        
        out << "}" << std::endl << std::endl;

        // jacobian
        out << "void eval_generated_jacobian(Kokkos::View<double **> &x, Kokkos::View<double **> &J, uint64_t n) {" << std::endl;

        for (int i = 0; i < total_params; i++) {
            out << "    double " + get_factor(i, stages) << " = x(" << std::to_string(i) << ", n);" << std::endl;
        }

        out << std::endl;

        int derivation_param = -1;
        for (int i = 0; i < jacobian_h.sizes.size(); i++) {
            uint32_t size = jacobian_h.sizes[i];
            uint32_t index = jacobian_h.indexes[i];

            if (i % (jacobian_h.sizes.size() / total_params) == 0) {
                derivation_param++;
                out << "    // derivating over " + get_factor(derivation_param, stages) << std::endl;
            }

            out << "    J(" << std::to_string(i) << ", n) = ";

            for (int k = 0; k < size; k += 1) {
                for (int l = 0; l < order; l++) {
                    uint8_t prod = jacobian_h.params(index + k, l);

                    out << get_factor(prod, stages);
                    if (l < order - 1) out << "*";
                }
                if (k < size - 1) out << " + ";
            }
            if (size == 0) out << "0";
            out << ";" << std::endl;
        }    
        
        out << "}" << std::endl;

        out.close();
    }
    Kokkos::finalize();
}