cmake_minimum_required(VERSION 3.25)
project(rk-playground LANGUAGES CXX)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos
    GIT_TAG 5.1.1
)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(
    Kokkos-kernels
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels
    GIT_TAG 5.1.1
)
FetchContent_MakeAvailable(Kokkos-kernels)

add_subdirectory(src)