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

# FetchContent_Declare(
#     Kokkos-tools
#     GIT_REPOSITORY https://github.com/kokkos/kokkos-tools
#     GIT_TAG develop
# )
# set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(Kokkos-tools)

FetchContent_Declare(
    json 
    URL 
    https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    fmt 
    GIT_REPOSITORY https://github.com/fmtlib/fmt 
    GIT_TAG 12.2.0
)
FetchContent_MakeAvailable(fmt)


add_subdirectory(src)