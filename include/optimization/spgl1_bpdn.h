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
	float min_step = 1e-16;
	float bp_tol = 1e-6;
	float ls_tol = 1e-6;
	float opt_tol = 1e-4;
	float dec_tol = 1e-4;
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
        std::vector<std::vector<float>>& A;
        std::vector<float>& b;
        float sigma;

        // Linalg Backend
        linalg_ops ops;

        // Timer Items
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop;

        // Running Params
        int rows;
        int cols;
        std::vector<float> init_x;
        float bnorm;
        std::vector<float> fvals;
        std::vector<float> xgrad;
        std::vector<float> ggrad;
        std::vector<std::vector<float>> trans;
        std::vector<float> At_flat;
        std::vector<float> A_flat;
        std::vector<float> x;
        std::vector<float> xold;
        std::vector<float> xbest;
        std::vector<float> r;
        std::vector<float> rold;
        std::vector<float> g;
        std::vector<float> gold;
        float f;
        float fbest;
        float fold;
        std::vector<float> dx;
        float dx_norm;
        std::vector<float> matvec_result;
	    std::vector<float> rmatvec_result;

        // Parameter and Iteration Information Structs
        params args;
        info iter_info;

        // Output
        std::vector<float> x_out;

        int verbosity;

        float fchange;
        float g_step;
        float gnorm, rnorm;
        float gap, rgap;
        float aerror1, aerror2;
        float rerror1, rerror2;

        // Tau Related Variabls
        float tau = 0.0;
        float tau_old;
        bool relchange1 = false;
        bool relchange2 = false;

        // Projected Gradient and Linesearch Vairables
        bool lnerr = false;
        float gts;
        float snorm = 0;
        float snormold = 0;
        float step = 1;
        float scale = 1;
        float gamma = 1e-4;
        int nsafe = 0;
        int n_iters = 0;
        float n, gtd;
        float tmp;
        std::vector<float> s;
        std::vector<float> b_search;

        spgl1_bpdn(std::vector<std::vector<float>>& A, std::vector<float>& b, float sigma, const params args = params(), linalg_backends backend = linalg_backends::CPU);
        void run(int max_iter = 100);
        void update_iterates();
        void update_tau();
        void compute_exit_conditions();
        bool check_exit_conditions();

        bool curve_line_search();
        bool dirn_line_search();

        void compute_gradient();
};