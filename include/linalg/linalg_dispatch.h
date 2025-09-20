#pragma once

#include <vector>
#include <functional>
#include <string>

enum class linalg_backends{
    CPU,
    OPENMP
};

// Function pointer types for all linalg operations
using matvec_func     = void(*)(const std::vector<float>&, const std::vector<float>&, int, int, std::vector<float>&);
using vecvec_func     = std::vector<float>(*)(const std::vector<float>&, const std::vector<float>&);
using scalarvec_func  = std::vector<float>(*)(const float&, const std::vector<float>&);
using norm_func       = float(*)(const std::vector<float>&);
using dot_func        = float(*)(const std::vector<float>&, const std::vector<float>&);
using projection_func = std::vector<float>(*)(std::vector<float>, float);

struct linalg_ops {
    matvec_func matvec;
    matvec_func rmatvec;

    vecvec_func vector_subtract;
    scalarvec_func scalar_vector_prod;

    norm_func l1_norm;
    norm_func l2_norm;
    norm_func inf_norm;

    dot_func dot;
    projection_func l1_norm_projection;
    
    std::string backend_name;
};

class linalg_dispatch{
    public:
        static linalg_ops get_ops(linalg_backends backend);
        static bool is_available(linalg_backends backend);
};