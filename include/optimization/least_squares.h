#pragma once

#include <vector>

#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include <linalg/linalg_dispatch.h>
#include <chrono>

enum class optimizers {
    SIMPLE_GRADIENT,
    CONJUGATE_GRADIENT
};

struct lsm_info {
    int n_iter = 0;
    int n_matvec = 0;
    int n_rmatvec = 0;
    float matvec_time = 0;
    float rmatvec_time = 0;
};

class least_squares_migration{
    public:
        least_squares_migration(std::vector<std::vector<float>>& L, std::vector<float>& d, linalg_backends backend = linalg_backends::OPENMP);

        void run(optimizers optimizer = optimizers::CONJUGATE_GRADIENT, int max_iterations = 50, float tol = 1e-6, int verbosity = 2);

        const std::vector<float>& get_model() const { return x_out; }
        float get_final_misfit() const { return final_misfit; }

    private:
        // Problem matrices and data
        std::vector<std::vector<float>>& A;  // Forward operator matrix
        std::vector<float>& b;               // Observed data
        std::vector<std::vector<float>> At;  // Transposed operator (adjoint)
        std::vector<float> A_flat;           // Flattened L for efficient matvec
        std::vector<float> At_flat;          // Flattened Lt for efficient rmatvec
        
        // Input Shapes
        int rows, cols;

        // Linalg Backend
        linalg_ops ops;

        // Iteration info
        lsm_info iter_info;
        std::chrono::high_resolution_clock::time_point start, stop;
        
        // Storage vectors
        std::vector<float> x_out;
        std::vector<float> g;
        std::vector<float> r;

        std::vector<float> matvec_result;
        std::vector<float> rmatvec_result;
        
        // Storage for BB Steps (SG) and Line Search (CG)
        std::vector<float> x_old;
        std::vector<float> g_old;
        std::vector<float> x_grad;
        std::vector<float> g_grad;

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

        // Step Size Computation Functions
        float compute_bb_step_size();
        float armijo_line_search(const std::vector<float>& direction, float initial_step = 1.0f);

        // Line Search Helper Method
        float evaluate_objective_at_step(const std::vector<float>& direction, float step_size);

        void run_simple_gradient(int max_iterations, float tol, int verbosity);
        void run_conjugate_gradient(int max_iterations, float tol, int verbosity);
        void precompute_operators();
};