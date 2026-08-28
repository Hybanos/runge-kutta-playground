set(executables main bench)

foreach(exec ${executables})

    add_executable(${exec})

    target_sources(${exec} PRIVATE 
        tree.cpp
        equations.cpp
        solve.cpp
        io.cpp

        ${exec}.cpp
    )

    include_directories(../include)

    # target_compile_options(main PRIVATE -g -march=native -funroll-loops -ftree-vectorize)
    target_compile_options(${exec} PRIVATE -g -march=native)

    target_link_libraries(${exec} PRIVATE Kokkos::kokkos Kokkos::kokkoskernels nlohmann_json::nlohmann_json fmt::fmt CLI11::CLI11)
    target_compile_features(${exec} PRIVATE cxx_std_20)

endforeach()