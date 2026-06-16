add_executable(main)

target_sources(main PRIVATE 
    main.cpp
)

include_directories(../include)

target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_23)