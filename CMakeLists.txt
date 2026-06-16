cmake_minimum_required(VERSION 3.25)
project(rk-playground)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos
    GIT_TAG 5.1.1 
)
FetchContent_MakeAvailable(Kokkos)

add_subdirectory(src)