#pragma once

#include <vector>
#include <chrono>
#include <linalg/linalg_dispatch.h>

// Optional Input Parameters
struct params {
	// Various Iteration Limits
	int iter_limit = 3000;
	int verbosity = 0;
	float max_step = 1e5;
	float min_step = 1e-16f;
	float bp_tol = 1e-6f;
	float ls_tol = 1e-6f;
	float opt_tol = 1e-4f;
	float dec_tol = 1e-4f;
};

struct info {
	// Iteration Information
	int n_iter = 0;
	int n_matvec = 0;
	int n_rmatvec = 0;
	int n_newton = 0;
	float matvec_time = 0;
	float rmatvec_time = 0;
	float proj_time = 0;
};

class spgl1_bpdn{
    public:
        //Input Variables
        std::vector<std::vector<float>>& A; // Forward operator matrix
        std::vector<float>& b;              // Observed data
        std::vector<std::vector<float>> At; // Transposed operator (adjoint)
        std::vector<float> At_flat;         // Flattened L for efficient matvec
        std::vector<float> A_flat;          // Flattened Lt for efficient rmatvec
        float sigma;

        // Input Shapes
        int rows, cols;

        // Linalg Backend
        linalg_ops ops;

        // Iteration Info
        info iter_info;
        std::chrono::high_resolution_clock::time_point start, stop;

        // Storage Vectors
        std::vector<float> x_out;
        std::vector<float> g;
        std::vector<float> r;

        std::vector<float> matvec_result;

        // Storage for Step Optimization
        std::vector<float> x_old;
        std::vector<float> g_old;
        std::vector<float> x_grad;
        std::vector<float> g_grad;

        float bnorm;
        std::vector<float> f_vals;
        std::vector<float> x_best;
        std::vector<float> r_old;
        
        float f;
        float f_best;
        float f_old
        ;
        std::vector<float> dx;
        float dx_norm;
        

        // Parameter and Iteration Information Structs
        params args;

        // Output
        int verbosity;

        // Exit Conditions
        float fchange;
        float g_step;
        float gnorm, rnorm;
        float gap, rgap;
        float aerror1, aerror2;
        float rerror1, rerror2;

        // Tau Related Variabls
        float tau = 50.0;
        float tau_old;
        bool relchange1 = false;
        bool relchange2 = false;

        // Projected Gradient and Linesearch Vairables
        float gts;
        float snorm = 0;
        float snormold = 0;
        float step = 1;
        float scale = 1;
        float gamma = 1e-4f;
        int nsafe = 0;
        int n_iters = 0;
        float n, gtd;
        float tmp;
        std::vector<float> s;
        std::vector<float> b_search;

        spgl1_bpdn(std::vector<std::vector<float>>& A, std::vector<float>& b, float sigma, const params args = params(), linalg_backends backend = linalg_backends::CUDA);
        void run(int max_iter = 100);
        void update_iterates();
        void update_tau();
        void compute_exit_conditions();
        bool check_exit_conditions();

        bool curve_line_search();
        bool dirn_line_search();

        void compute_gradient();
        void set_optimal_params();
        void precompute_operators();
        void compute_residual();
};