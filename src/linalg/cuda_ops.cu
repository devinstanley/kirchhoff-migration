#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace cuda_ops {

__global__ void matvec_kernel(const float* A, const float* x, float* y, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float temp = 0.0f;
        for (int j = 0; j < n; ++j) {
            temp += A[row * n + j] * x[j];
        }
        y[row] = temp;
    }
}

void matvec(const std::vector<float>& mat,
            const std::vector<float>& vec,
            int m, int n,
            std::vector<float>& res) {
    float *d_A, *d_x, *d_y;

    cudaMalloc(&d_A, mat.size() * sizeof(float));
    cudaMalloc(&d_x, vec.size() * sizeof(float));
    cudaMalloc(&d_y, res.size() * sizeof(float));

    cudaMemcpy(d_A, mat.data(), mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;
    matvec_kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y, m, n);
    cudaDeviceSynchronize();

    cudaMemcpy(res.data(), d_y, res.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

std::vector<float> vector_subtract(const std::vector<float>& a, const std::vector<float>& b) {
    thrust::device_vector<float> d_a = a;
    thrust::device_vector<float> d_b = b;
    thrust::device_vector<float> d_res(a.size());

    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_res.begin(), thrust::minus<float>());

    std::vector<float> res(a.size());
    thrust::copy(d_res.begin(), d_res.end(), res.begin());
    return res;
}

std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec) {
    thrust::device_vector<float> d_vec = vec;
    thrust::device_vector<float> d_res(vec.size());

    thrust::transform(d_vec.begin(), d_vec.end(), d_res.begin(),
                      [scalar] __device__ (float x) { return scalar * x; });

    std::vector<float> res(vec.size());
    thrust::copy(d_res.begin(), d_res.end(), res.begin());
    return res;
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    thrust::device_vector<float> d_a = a;
    thrust::device_vector<float> d_b = b;

    return thrust::inner_product(d_a.begin(), d_a.end(), d_b.begin(), 0.0f);
}

float l1_norm(const std::vector<float>& vec) {
    thrust::device_vector<float> d_vec = vec;
    return thrust::transform_reduce(d_vec.begin(), d_vec.end(),
                                    [] __device__ (float x) { return fabsf(x); },
                                    0.0f, thrust::plus<float>());
}

float l2_norm(const std::vector<float>& vec) {
    thrust::device_vector<float> d_vec = vec;
    float sum_squares = thrust::transform_reduce(d_vec.begin(), d_vec.end(),
                                                 [] __device__ (float x) { return x * x; },
                                                 0.0f, thrust::plus<float>());
    return std::sqrt(sum_squares);
}

float inf_norm(const std::vector<float>& vec) {
    thrust::device_vector<float> d_vec = vec;
    return thrust::transform_reduce(d_vec.begin(), d_vec.end(),
                                    [] __device__ (float x) { return fabsf(x); },
                                    0.0f, thrust::maximum<float>());
}

std::vector<float> l1_norm_projection(std::vector<float> vec, float tau) {
    return vec;
}

}
#endif