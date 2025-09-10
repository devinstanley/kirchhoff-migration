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

        void run(const std::vector<float>& data, optimizers optimizer = optimizers::CONJUGATE_GRADIENT, int max_iterations = 50, float tol = 1e-6, int verbosity = 2);

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

        // Storage for BB Steps (SG) and Line Search (CG)
        std::vector<float> prev_model;
        std::vector<float> prev_gradient;
        std::vector<float> model_step;
        std::vector<float> gradient_step;

        float final_misfit;

        const float DEFAULT_STEP_SIZE = 1e-6f;
        const float BETA_ZERO_THRESHOLD = 1e-15f;

        // Line search parameters
        const float ARMIJO_C1 = 1e-2f;          // Armijo Condition
        const float BACKTRACK_FACTOR = 0.5f;    // Backtracking Factor
        const float MAX_STEP_SIZE = 1.0f;       // Maximum Step Size
        const float MIN_STEP_SIZE = 1e-6f;     // Minimum Step Size
        const int MAX_LINE_SEARCH_ITER = 20;    // Maximum # Line Search Iterations

        void compute_gradient();
        void compute_residual();
        float compute_misfit();
        float dot_product(const std::vector<float>& a, const std::vector<float>& b);

        // Step Size Computation Functions
        float compute_bb_step_size();
        float armijo_line_search(const std::vector<float>& direction, float initial_step = 1.0f);

        // Line Search Helper Method
        float evaluate_objective_at_step(const std::vector<float>& direction, float step_size);

        void run_simple_gradient(int max_iterations, float tol, int verbosity);
        void run_conjugate_gradient(int max_iterations, float tol, int verbosity);
};