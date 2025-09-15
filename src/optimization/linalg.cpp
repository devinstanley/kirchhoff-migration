#include "linalg.h"
#include <math.h>
#include <numeric>
#include <algorithm>

std::vector<float> linalg::l1_norm_projection(std::vector<float> vec, float tau){
    int n = vec.size();
	std::vector<float> proj(n, 0);
	std::vector<float> alpha(n + 1, 0);
	std::vector<float> sum_b(n, 0);
	std::vector<int> sgn(n, 1);
	int alpha_idx = -1;
	float alpha_prev;
	float norm = l1_norm(vec);
	float cur_sum = 0;


	//Create Index Vector
	std::vector<float> idx(n);
	std::iota(idx.begin(), idx.end(), 0);

	//Exit Early
	if (tau >= norm) {
		return vec;
	}
	else if (tau < std::numeric_limits<float>::epsilon()) {
		return proj;
	}
	else {
		//Take Absolute Value of Input
		for (int ii = 0; ii < n; ii++) {
			if (vec[ii] < 0) {
				sgn[ii] = -1;
			}
			vec[ii] = abs(vec[ii]);
		}
		//Reverse Argsort
		std::stable_sort(idx.begin(), idx.end(),
			[&vec](size_t i1, size_t i2) {return vec[i1] < vec[i2]; });
		std::reverse(idx.begin(), idx.end());

		//Reverse Sort B
		std::sort(vec.begin(), vec.end(), std::greater<>());

		for (int i = 0; i < n; i++) {
			cur_sum += vec[i];
			sum_b[i] = cur_sum - tau;
		}

		for (int i = 1; i < n + 1; i++) {
			alpha[i] = sum_b[i - 1] / i;
		}
		for (int i = 0; i < n; i++) {
			if (alpha[i + 1] >= vec[i]) {
				alpha_idx = i;
				break;
			}
		}
		if (alpha_idx >= 0) {
			alpha_prev = alpha[alpha_idx];
		}
		else {
			alpha_prev = alpha[alpha.size()];
		}

		for (int i = 0; i < n; i++) {
			proj[idx[i]] = vec[i] - alpha_prev;
			if (proj[idx[i]] < 0) {
				proj[idx[i]] = 0;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		proj[i] *= sgn[i];
	}

	return proj;
}
float linalg::l1_norm(const std::vector<float>& vec){
	const float* __restrict__ x = vec.data();
    float norm = 0.0f;
	
	#pragma omp simd reduction(+:norm)
    for (int i = 0; i < (int)vec.size(); ++i) {
        norm += std::fabs(x[i]);  // use fabsf for float
    }
    return norm;
}
float linalg::l2_norm(const std::vector<float>& vec){
    const float* __restrict__ x = vec.data();
    float norm = 0.0f;

    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < (int)vec.size(); ++i) {
        norm += x[i] * x[i];
    }
    return std::sqrt(norm);
}
float linalg::inf_norm(const std::vector<float>& vec){
    const float* __restrict__ x = vec.data();
    float maxv = 0.0f;

    #pragma omp simd reduction(max:maxv)
    for (int i = 0; i < (int)vec.size(); ++i) {
        float v = std::fabs(x[i]);
        if (v > maxv) maxv = v;
    }
    return maxv;
}

std::vector<float> linalg::scalar_vector_prod(const float& scalar, const std::vector<float>& vec){
    int n = (int)vec.size();
    std::vector<float> prod(n);
    const float* __restrict__ x = vec.data();
    float* __restrict__ y = prod.data();

    #pragma omp simd
    for (int i = 0; i < n; ++i) {
        y[i] = scalar * x[i];
    }
    return prod;
}
std::vector<float> linalg::vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2){
    int n = (int)vec1.size();
    std::vector<float> sub(n);
    const float* __restrict__ x1 = vec1.data();
    const float* __restrict__ x2 = vec2.data();
    float* __restrict__ y = sub.data();

    #pragma omp simd
    for (int i = 0; i < n; ++i) {
        y[i] = x1[i] - x2[i];
    }
    return sub;
}
float linalg::dot(const std::vector<float>& vec1, const std::vector<float>& vec2){
    const float* __restrict__ x1 = vec1.data();
    const float* __restrict__ x2 = vec2.data();
    float sum = 0.0f;

    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < (int)vec1.size(); ++i) {
        sum += x1[i] * x2[i];
    }
    return sum;
}

void linalg::matvec(const std::vector<float>& mat,
                    const std::vector<float>& vec,
                    int m, int n,
                    std::vector<float>& res) {
    const float* __restrict__ A = mat.data();
    const float* __restrict__ x = vec.data();
    float* __restrict__ y = res.data();

    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        float temp = 0.0f;
        const float* row = A + i * n;

        #pragma omp simd reduction(+:temp)
        for (int j = 0; j < n; ++j) {
            temp += row[j] * x[j];
        }
        y[i] = temp;
    }
}