#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>

#include <Kokkos_Core.hpp>

template<class T>
void save_scalar(std::string path, T v) {
    std::ofstream f;
    f.open(path, std::ios::out | std::ios::binary);
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
    f.close();
}

template<class T>
T load_scalar(std::string path) {
    std::fstream f;
    T out;
    f.open(path);
    f.read((char *) &out, sizeof(T));
    f.close();
    return out;
}

template<class ViewType>
void save_view(std::string path, ViewType &v) {
    std::ofstream f;
    uint64_t size = 1;
    for (int i = 0; i < ViewType::rank; i++) size *= v.extent(i); 
    f.open(path, std::ios::out | std::ios::binary);
    for (uint64_t i = 0; i < size; i++) {
        f.write(reinterpret_cast<const char *>(&v.data()[i]), sizeof(typename ViewType::value_type));
    }
    f.close();
}

template<class ViewType, typename... Extents>
ViewType load_view(std::string path, Extents &&... extents) {
    std::fstream f;
    uint64_t size = std::filesystem::file_size(path);

    ViewType out(path, std::forward<Extents>(extents)...);

    f.open(path);
    f.read((char *) out.data(), size);
    f.close();

    return out;
}