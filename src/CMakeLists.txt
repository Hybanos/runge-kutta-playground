include_directories(../include)

add_executable(generator)

target_sources(generator PRIVATE 
    tree.cpp
    equations.cpp

    system_kernel_gen.cpp
)

target_compile_options(generator PRIVATE -g -march=native -ffast-math)
target_link_libraries(generator PRIVATE Kokkos::kokkos Kokkos::kokkoskernels nlohmann_json::nlohmann_json fmt::fmt)

target_compile_features(generator PRIVATE cxx_std_20)

set(GENERATED_CPP "${CMAKE_CURRENT_BINARY_DIR}/gen/p3_s3.cpp")

add_custom_command(
    OUTPUT "${GENERATED_CPP}"
    COMMAND $<TARGET_FILE:generator> 3 3
    DEPENDS generator
    VERBATIM
)

set(executables main bench)

foreach(exec ${executables})

    add_executable(${exec})

    target_sources(${exec} PUBLIC
        ${GENERATED_CPP}
    )
    target_sources(${exec} PRIVATE 
        tree.cpp
        equations.cpp
        solve.cpp
        io.cpp

        ${exec}.cpp
    )

    # target_compile_options(main PRIVATE -g -march=native -funroll-loops -ftree-vectorize)
    target_compile_options(${exec} PRIVATE -g -march=native)

    target_link_libraries(${exec} PRIVATE Kokkos::kokkos Kokkos::kokkoskernels nlohmann_json::nlohmann_json fmt::fmt CLI11::CLI11)
    target_compile_features(${exec} PRIVATE cxx_std_20)

endforeach()