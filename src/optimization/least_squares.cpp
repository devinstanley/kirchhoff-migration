#include <iostream>
#include "least_squares.h"
#include <limits>
#include <numeric>


least_squares_migration::least_squares_migration(
    std::vector<std::vector<float>>& A, 
    std::vector<float>& b, linalg_backends backend):
    A(A), b(b)
{
    if (A.empty() || A[0].empty()) {
        throw std::invalid_argument("Forward operator b cannot be empty!");
    }
    if (b.empty()) {
        throw std::invalid_argument("Data vector b cannot be empty!");
    }

    iter_info = lsm_info();
    rows = (int)A.size();
    cols = (int)A[0].size();

    
    if (b.size() != rows) {
        throw std::invalid_argument("Data size mismatch with operator dimensions!");
    }

    ops = linalg_dispatch::get_ops(backend);
    
    // Initialize Main Storage Space
    x_out.assign(cols, 0.0f);
    g.assign(cols, 0.0f);
    r.assign(rows, 0.0f);
    matvec_result.assign(rows, 0.0f);
    
    // Initialize step size storage
    x_old.assign(cols, 0.0f);
    g_old.assign(cols, 0.0f);
    x_grad.assign(cols, 0.0f);
    g_grad.assign(cols, 0.0f);
    
    // Precompute Flattened Operators
    precompute_operators();
}

void least_squares_migration::precompute_operators() {
    // Precompute Raveled A and A_T
    A_flat.reserve(rows * cols);
    At_flat.reserve(rows * cols);
    At.assign(cols, std::vector<float>(rows));
    
    // Flatten A
    for (const auto& row : A) {
        A_flat.insert(A_flat.end(), row.begin(), row.end());
    }
    
    // Compute Transpose
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            At[j][i] = A[i][j];
        }
    }
    // Flatten Transpose
    for (const auto& row : At) {
        At_flat.insert(At_flat.end(), row.begin(), row.end());
    }
}

void least_squares_migration::run(optimizers optimizer, int max_iterations, float tol, int verbosity){
    if (max_iterations <= 0) {
        throw std::invalid_argument("Max iterations must be positive!");
    }

    iter_info = lsm_info();

    switch (optimizer){
        case optimizers::SIMPLE_GRADIENT:
            run_simple_gradient(max_iterations, tol, verbosity);
            break;
        case optimizers::CONJUGATE_GRADIENT:
            run_conjugate_gradient(max_iterations, tol, verbosity);
            break;
    }

    final_misfit = compute_misfit();
    std::cout << "Final Misfit: " << final_misfit << std::endl;
}

void least_squares_migration::run_simple_gradient(int max_iterations, float tol, int verbosity){
    float alpha = DEFAULT_STEP_SIZE;
    float prev_misfit = std::numeric_limits<float>::max();

    // Compute Initial Gradient
    compute_gradient();

    for (int iter = 0; iter < max_iterations; iter++){
        iter_info.n_iter += 1;

        // Store Previous Models for BB Step
        x_old = x_out;
        g_old = g;

        // Update Model
        x_out = ops.vector_subtract(x_old, ops.scalar_vector_prod(alpha, g));

        // Compute and Store Gradient in Vector
        compute_gradient();

        // Compute BB Step
        if (iter != 0){
            alpha = compute_bb_step_size();
        }

        // Compute Misfit
        float current_misfit = compute_misfit();
        float misfit_reduction = prev_misfit - current_misfit;

        if (verbosity > 1 || (verbosity > 0 && iter % 10 == 0)) {
            std::cout << iter << "\t" << current_misfit << "\t" 
                     << misfit_reduction << "\t" << alpha << std::endl;
        }
        
        // Check convergence
        if (std::abs(misfit_reduction) < tol && iter > 0) {
            std::cout << "Converged after " << iter << " iterations" << std::endl;
            break;
        }

        prev_misfit = current_misfit;
    }
}

float least_squares_migration::compute_bb_step_size(){
    // Compute Model Step
    x_grad = ops.vector_subtract(x_out, x_old);

    // Compute Gradient Step
    g_grad = ops.vector_subtract(g, g_old);

    // Compute BB Step Sizes
    float s_dot_y = ops.dot(x_grad, g_grad);
    float s_dot_s = ops.dot(x_grad, x_grad);
    float y_dot_y = ops.dot(g_grad, g_grad);

    float alpha_bb1, alpha_bb2, alpha;
    
    // BB1: alpha = s^T s / s^T y
    if (std::abs(s_dot_y) > 1e-15f) {
        alpha_bb1 = s_dot_s / s_dot_y;
    } else {
        alpha_bb1 = DEFAULT_STEP_SIZE;
    }
    
    // BB2: alpha = s^T y / y^T y  
    if (y_dot_y > 1e-15f) {
        alpha_bb2 = s_dot_y / y_dot_y;
    } else {
        alpha_bb2 = DEFAULT_STEP_SIZE;
    }
    
    // Choose BB1 or BB2 if reasonable
    if (alpha_bb2 > 0 && alpha_bb2 < 1e6 * DEFAULT_STEP_SIZE) {
        alpha = alpha_bb2;
    } else if (alpha_bb1 > 0 && alpha_bb1 < 1e6 * DEFAULT_STEP_SIZE) {
        alpha = alpha_bb1;
    } else {
        alpha = DEFAULT_STEP_SIZE;
    }
    
    // Safeguard
    alpha = std::max(MIN_STEP_SIZE, std::min(alpha, MAX_STEP_SIZE));
    
    return alpha;
}

void least_squares_migration::run_conjugate_gradient(int max_iterations, float tol, int verbosity){
    float alpha;
    float prev_misfit = std::numeric_limits<float>::max();
    float beta, r_norm, r_prev_norm;

    //Initial Gradient Computation
    compute_gradient();

    // Initialize Conjugate Gradient Direction
    std::vector<float> conj_dir = ops.scalar_vector_prod(-1.0, g);

    for (int iter = 0; iter < max_iterations; iter++) {
        iter_info.n_iter += 1;

        // Store Previous Gradient
        g_old = g;

        // Line Search for Optimal Step Size
        alpha = armijo_line_search(conj_dir, 1e-6f);

        // Update Model
        #pragma omp parallel for
        for (int i = 0; i < x_out.size(); i++) {
            x_out[i] += alpha * conj_dir[i];
        }

        // Compute New Gradient
        compute_gradient();

        // Fletcher-Reeves Formula
        r_norm = ops.dot(g, g);
        r_prev_norm = ops.dot(g_old, g_old);

        // Zero Division Check
        if (r_prev_norm < 1e-15f) {
            beta = 0.0f;
        } else {
            beta = r_norm / r_prev_norm;
        }

        // Update Conjugate Direction
        #pragma omp parallel for
        for (int i = 0; i < conj_dir.size(); i++) {
            conj_dir[i] = -g[i] + beta * conj_dir[i];
        }

        // Compute Misfit
        float current_misfit = compute_misfit();
        float misfit_reduction = prev_misfit - current_misfit;

        if (verbosity > 1 || (verbosity > 0 && iter % 10 == 0)) {
            std::cout << iter << "\t" << current_misfit << "\t" 
                     << misfit_reduction << "\t" << beta << "\t\t" << alpha << std::endl;
        }
        
        // Check Convergence
        if (std::abs(misfit_reduction) < tol && iter > 0) {
            std::cout << "Converged after " << iter << " iterations" << std::endl;
            break;
        }

        prev_misfit = current_misfit;
    }
}

float least_squares_migration::armijo_line_search(const std::vector<float>& direction, float initial_step) {
    float current_misfit = compute_misfit();
    float directional_derivative = ops.dot(g, direction);
    
    // Direction Not in Descent, Return Min Step
    if (directional_derivative >= 0) {
        return MIN_STEP_SIZE;
    }
    
    float step = initial_step;
    
    for (int i = 0; i < MAX_LINE_SEARCH_ITER; i++) {
        float new_misfit = evaluate_objective_at_step(direction, step);
        
        // Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*g^T*d
        if (new_misfit <= current_misfit + ARMIJO_C1 * step * directional_derivative) {
            return step;
        }
        
        step *= BACKTRACK_FACTOR;
        
        // If step becomes too small, return it
        if (step < MIN_STEP_SIZE) {
            return MIN_STEP_SIZE;
        }
    }
    
    return step;
}

float least_squares_migration::evaluate_objective_at_step(const std::vector<float>& direction, float step_size) {
    // Save Current Model
    std::vector<float> original_model = x_out;
    
    // Take Step
    #pragma omp parallel for
    for (int i = 0; i < x_out.size(); i++) {
        x_out[i] += step_size * direction[i];
    }
    
    // Evaluate Misfit
    float objective = compute_misfit();
    
    // Restore Original
    x_out = original_model;
    
    return objective;
}

void least_squares_migration::compute_gradient(){
    // Forward
    start = std::chrono::high_resolution_clock::now();
    ops.matvec(A_flat, x_out, rows, cols, matvec_result);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_matvec += 1;
    
    // Compute Residual
    compute_residual();
    
    // Adjoint
    start = std::chrono::high_resolution_clock::now();
    ops.matvec(At_flat, r, cols, rows, g);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_rmatvec += 1;
}

void least_squares_migration::compute_residual(){
    r = ops.vector_subtract(matvec_result, b);
}

float least_squares_migration::compute_misfit() {
    float misfit = 0.0f;
    compute_residual();

    misfit = ops.dot(r, r);
    
    return 0.5f * misfit;
}