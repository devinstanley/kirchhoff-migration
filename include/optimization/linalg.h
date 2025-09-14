#pragma once

#include <vector>

class linalg{
    public:
        static std::vector<float> l1_norm_projection(std::vector<float> vec, float tau);
        static float l1_norm(const std::vector<float>& vec);
        static float l2_norm(const std::vector<float>& vec);
        static float inf_norm(const std::vector<float>& vec);

        static std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec);
        static std::vector<float> vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2);
        static float dot(const std::vector<float>& vec1, const std::vector<float>& vec2);
        static void matvec(std::vector<float> const& mat, std::vector<float> const& vec, int m, int n, std::vector<float>& res);
};