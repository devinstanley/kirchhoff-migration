#pragma once
#include "seismic_model.h"
#include <vector>

class forward_kirchhoff{
    public:
        seismic_model& env;
        std::vector<float> d;
        std::vector<std::vector<float>> L;

        forward_kirchhoff(seismic_model& env);

        void run();
        void d_to_file(const std::string& path);
        void L_to_file(const std::string& path);
};