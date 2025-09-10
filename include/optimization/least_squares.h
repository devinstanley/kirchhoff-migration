#pragma once

#include <vector>

#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"

class least_squares_migration{
    public:
        enum class optimizers{
            SIMPLE_GRADIENT,
            CONJUGATE_GRADIENT
        };
        least_squares_migration(seismic_model& env);

        void run(const std::vector<float>& data, optimizers optimizer = optimizers::CONJUGATE_GRADIENT, int max_iterations = 50, float tol = 1e-6, int verbosity = 1);

        const std::vector<float>& get_model() const { return model; }
        float get_final_misfit() const { return final_misfit; }

    private:
        seismic_model& env;
        forward_kirchhoff fwd;
        adjoint_kirchhoff adj;
        
        const std::vector<float>* data_ptr;
        std::vector<float> model;
        std::vector<float> gradient;
        std::vector<float> residual;
        std::vector<float> predicted_data;

        float final_misfit;

        const float DEFAULT_STEP_SIZE = 1e-6f;
        const float BETA_ZERO_THRESHOLD = 1e-15f;

        void compute_gradient();
        void compute_residual();
        float compute_misfit();
        float dot_product(const std::vector<float>& a, const std::vector<float>& b);

        void run_simple_gradient(int max_iterations, float tol, int verbosity);
        void run_conjugate_gradient(int max_iterations, float tol, int verbosity);
};