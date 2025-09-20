#pragma once
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

namespace cuda_ops {

// matvec kernel declaration
__global__ void matvec_kernel(const float* A, const float* x, float* y, int m, int n);

// Host function declarations
void matvec(const std::vector<float>& mat,
            const std::vector<float>& vec,
            int m, int n,
            std::vector<float>& res);

std::vector<float> vector_subtract(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec);
float dot(const std::vector<float>& a, const std::vector<float>& b);
float l1_norm(const std::vector<float>& vec);
float l2_norm(const std::vector<float>& vec);
float inf_norm(const std::vector<float>& vec);

std::vector<float> l1_norm_projection(std::vector<float> vec, float tau);

} // namespace cuda_ops
#endif