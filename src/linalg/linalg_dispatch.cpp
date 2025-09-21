#include "linalg_dispatch.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef HAVE_CUDA
#include "cuda_ops.cuh"
#endif
#include <numeric>
#include <queue>

// CPU Only Implementations
namespace cpu_ops {
    void matvec(const std::vector<float>& mat,
                        const std::vector<float>& vec,
                        int m, int n,
                        std::vector<float>& res) {
        const float* __restrict A = mat.data();
        const float* __restrict x = vec.data();
        float* __restrict y = res.data();

        for (int i = 0; i < m; ++i) {
            float temp = 0.0f;
            const float* row = A + i * n;

            for (int j = 0; j < n; ++j) {
                temp += row[j] * x[j];
            }
            y[i] = temp;
        }
    }

    std::vector<float> vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2){
        int n = (int)vec1.size();
        std::vector<float> sub(n);
        const float* __restrict x1 = vec1.data();
        const float* __restrict x2 = vec2.data();
        float* __restrict y = sub.data();

        for (int i = 0; i < n; ++i) {
            y[i] = x1[i] - x2[i];
        }
        return sub;
    }

    std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec){
        int n = (int)vec.size();
        std::vector<float> prod(n);
        const float* __restrict x = vec.data();
        float* __restrict y = prod.data();

        for (int i = 0; i < n; ++i) {
            y[i] = scalar * x[i];
        }
        return prod;
    }

    float l1_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float norm = 0.0f;
        
        for (int i = 0; i < (int)vec.size(); ++i) {
            norm += std::fabs(x[i]);  // use fabsf for float
        }
        return norm;
    }
    float l2_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float norm = 0.0f;

        for (int i = 0; i < (int)vec.size(); ++i) {
            norm += x[i] * x[i];
        }
        return std::sqrt(norm);
    }
    float inf_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float maxv = 0.0f;

        for (int i = 0; i < (int)vec.size(); ++i) {
            float v = std::fabs(x[i]);
            if (v > maxv) maxv = v;
        }
        return maxv;
    }

    float dot(const std::vector<float>& vec1, const std::vector<float>& vec2){
        const float* __restrict x1 = vec1.data();
        const float* __restrict x2 = vec2.data();
        float sum = 0.0f;

        for (int i = 0; i < (int)vec1.size(); ++i) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }

    std::vector<float> l1_norm_projection(std::vector<float>& vec, float tau){
        // Quick Exit Check Before Allocating Space
        float norm = l1_norm(vec);
        if (tau >= norm) {
            return vec;
        }

        // Check For Valid Tau
        int n = (int)vec.size();
        if (tau < std::numeric_limits<float>::epsilon()) {
            return std::vector<float>(n, 0);
        }

        const float* __restrict x = vec.data();
        std::priority_queue<float> heap;
        for (int i = 0; i < n; i++){
            heap.push(std::fabs(x[i]));
        }

        float gamma = 0.0f;
        float delta = 0.0f;
        float nu = -tau;
        float cmin;

        for (int i = 0; i < n; i++){
            cmin = heap.top();
            nu += cmin;
            gamma = nu / (i + 1);

            if (gamma >= cmin){
                break;
            }

            heap.pop();
            delta = gamma;
        }

        // Soft-threshold
        std::vector<float> proj(n, 0.0);
        float* __restrict y = proj.data();
        for (int i = 0; i < n; i++){
            y[i] = std::copysign(std::max(std::fabs(y[i]) - delta, 0.0f), y[i]);
        }

        return proj;
    }
}

#ifdef HAVE_OPENMP
// OpenMP Implementations
namespace openmp_ops {
    void matvec(const std::vector<float>& mat,
                        const std::vector<float>& vec,
                        int m, int n,
                        std::vector<float>& res) {
        const float* __restrict A = mat.data();
        const float* __restrict x = vec.data();
        float* __restrict y = res.data();

        #pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            float temp = 0.0f;
            const float* row = A + i * n;

            //#pragma omp simd reduction(+:temp)
            for (int j = 0; j < n; ++j) {
                temp += row[j] * x[j];
            }
            y[i] = temp;
        }
    }

    std::vector<float> vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2){
        int n = (int)vec1.size();
        std::vector<float> sub(n);
        const float* __restrict x1 = vec1.data();
        const float* __restrict x2 = vec2.data();
        float* __restrict y = sub.data();

        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            y[i] = x1[i] - x2[i];
        }
        return sub;
    }

    std::vector<float> scalar_vector_prod(const float& scalar, const std::vector<float>& vec){
        int n = (int)vec.size();
        std::vector<float> prod(n);
        const float* __restrict x = vec.data();
        float* __restrict y = prod.data();

        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            y[i] = scalar * x[i];
        }
        return prod;
    }

    float l1_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float norm = 0.0f;
        
        //#pragma omp simd reduction(+:norm)
        for (int i = 0; i < (int)vec.size(); ++i) {
            norm += std::fabs(x[i]);  // use fabsf for float
        }
        return norm;
    }
    float l2_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float norm = 0.0f;

        //#pragma omp simd reduction(+:norm)
        for (int i = 0; i < (int)vec.size(); ++i) {
            norm += x[i] * x[i];
        }
        return std::sqrt(norm);
    }
    float inf_norm(const std::vector<float>& vec){
        const float* __restrict x = vec.data();
        float maxv = 0.0f;

        for (int i = 0; i < (int)vec.size(); ++i) {
            maxv = std::max(maxv, std::fabs(x[i]));
        }
        return maxv;
    }

    float dot(const std::vector<float>& vec1, const std::vector<float>& vec2){
        const float* __restrict x1 = vec1.data();
        const float* __restrict x2 = vec2.data();
        float sum = 0.0f;

        //#pragma omp simd reduction(+:sum)
        for (int i = 0; i < (int)vec1.size(); ++i) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }
}
#endif

// Dispatcher Implementation
linalg_ops linalg_dispatch::get_ops(linalg_backends backend){
    linalg_ops ops;

    switch (backend) {
        case linalg_backends::CPU:
            ops.matvec = cpu_ops::matvec;
            ops.rmatvec = cpu_ops::matvec; // Same function, different matrix
            ops.vector_subtract = cpu_ops::vector_subtract;
            ops.scalar_vector_prod = cpu_ops::scalar_vector_prod;
            ops.l1_norm = cpu_ops::l1_norm;
            ops.l2_norm = cpu_ops::l2_norm;
            ops.inf_norm = cpu_ops::inf_norm;
            ops.dot = cpu_ops::dot;
            ops.l1_norm_projection = cpu_ops::l1_norm_projection;
            ops.backend_name = "CPU";
            break;

#ifdef HAVE_OPENMP
        case linalg_backends::OPENMP:
            ops.matvec = openmp_ops::matvec;
            ops.rmatvec = openmp_ops::matvec;
            ops.vector_subtract = openmp_ops::vector_subtract;
            ops.scalar_vector_prod = openmp_ops::scalar_vector_prod;
            ops.l1_norm = openmp_ops::l1_norm;
            ops.l2_norm = openmp_ops::l2_norm;
            ops.inf_norm = openmp_ops::inf_norm;
            ops.dot = openmp_ops::dot;
            ops.l1_norm_projection = cpu_ops::l1_norm_projection;
            ops.backend_name = "OpenMP";
            break;
#endif
#ifdef HAVE_CUDA
        case linalg_backends::CUDA:
            ops.matvec = cuda_ops::matvec;
            ops.rmatvec = cuda_ops::matvec;
            ops.vector_subtract = cuda_ops::vector_subtract;
            ops.scalar_vector_prod = cuda_ops::scalar_vector_prod;
            ops.l1_norm = cuda_ops::l1_norm;
            ops.l2_norm = cuda_ops::l2_norm;
            ops.inf_norm = cuda_ops::inf_norm;
            ops.dot = cuda_ops::dot;
            ops.l1_norm_projection = cpu_ops::l1_norm_projection;
            ops.backend_name = "CUDA";
            break;
#endif
        default:
            return get_ops(linalg_backends::CPU);
    }

    return ops;
}

bool linalg_dispatch::is_available(linalg_backends backend) {
    switch (backend) {
        case linalg_backends::CPU:
            return true;
            
        case linalg_backends::OPENMP:
#ifdef HAVE_OPENMP
            return true;
#else
            return false;
#endif
        case linalg_backends::CUDA:
#ifdef HAVE_CUDA
            return true;
#else
            return false;
#endif
    }
    return false;
}