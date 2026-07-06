add_executable(main)

target_sources(main PRIVATE 
    tree.cpp
    equations.cpp
    solve.cpp

    main.cpp
)

include_directories(../include)

target_compile_options(main PRIVATE -g -march=native -funroll-loops -ftree-vectorize)

target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_20)