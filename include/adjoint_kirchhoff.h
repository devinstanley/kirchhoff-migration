#pragma once

#include "seismic_model.h"
#include <vector>
#include <string>

class adjoint_kirchhoff {
public:
	seismic_model& env;
	std::vector<float> mig;

	adjoint_kirchhoff(seismic_model& env);
	void run(std::vector<float> d);
	void mig_to_file(const std::string& filename);
};