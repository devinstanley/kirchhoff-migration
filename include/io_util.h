#pragma once

#include <vector>
#include <string>
class io_util{
    public:
        static std::vector<double> import_vector_csv(const std::string& path);
        static std::vector<std::vector<double>> import_matrix_csv(const std::string& path);
};