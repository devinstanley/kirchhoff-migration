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
    float norm = 0;
	for (float value : vec) {
		norm += abs(value);
	}
	return norm;
}
float linalg::l2_norm(const std::vector<float>& vec){
    float norm = 0;
	for (float value : vec) {
		norm += value * value;
	}
	return sqrt(norm);
}
float linalg::inf_norm(const std::vector<float>& vec){
    float max = 0;
	for (float value : vec) {
		if (abs(value) > max) {
			max = abs(value);
		}
	}
	return max;
}

std::vector<float> linalg::scalar_vector_prod(const float& scalar, const std::vector<float>& vec){
    int n = vec.size();
	std::vector<float> prod(n, 0.0);

	for (int i = 0; i < n; i++) {
		prod[i] = scalar * vec[i];
	}
	return prod;
}
std::vector<float> linalg::vector_subtract(const std::vector<float>& vec1, const std::vector<float>& vec2){
    int n = vec1.size();
	std::vector<float> sub(n, 0.0);

	for (int i = 0; i < n; i++) {
		sub[i] = vec1[i] - vec2[i];
	}
	return sub;
}
float linalg::dot(const std::vector<float>& vec1, const std::vector<float>& vec2){
    float prod_sum = 0;
	for (int i = 0; i < vec1.size(); i++) {
		prod_sum += vec1[i] * vec2[i];
	}
	return prod_sum;
}

void linalg::matvec(std::vector<float> const& mat, std::vector<float> const& vec, int m, int n, std::vector<float>& res) {

	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		float temp = 0;
		for (int j = 0; j < n; ++j) {
			temp += mat[i * n + j] * vec[j];
		}
		res[i] = temp;
	}
}