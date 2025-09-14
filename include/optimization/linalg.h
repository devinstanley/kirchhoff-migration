#pragma once

#include <vector>

class linalg{
    public:
        std::vector<float> l1_norm_projection(std::vector<float> vec, float tau);
        float l1_norm(const std::vector<float>& vec);
        float l2_norm(const std::vector<float>& vec);
        float inf_norm(const std::vector<float>& vec);

        std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec);
        std::vector<float> vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2);
        float dot(const std::vector<float>& vec1, const std::vector<float>& vec2);
};