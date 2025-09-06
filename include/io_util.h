#pragma once

#include <vector>
#include <string>
class io_util{
    public:
        static std::vector<float> import_vector_csv(const std::string& path);
        static std::vector<std::vector<float>> import_matrix_csv(const std::string& path);
};