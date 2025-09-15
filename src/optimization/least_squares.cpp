#include <iostream>

#include "least_squares.h"
#include "linalg.h"
#include <limits>
#include <numeric>


least_squares_migration::least_squares_migration(
    std::vector<std::vector<float>>& L, 
    std::vector<float>& d
    ) : L(L), d(d) {
    
    if (L.empty() || L[0].empty()) {
        throw std::invalid_argument("Forward operator L cannot be empty!");
    }
    if (d.empty()) {
        throw std::invalid_argument("Data vector d cannot be empty!");
    }
    
    rows = L.size();
    cols = L[0].size();
    data_size = rows;
    model_size = cols;
    
    if (d.size() != rows) {
        throw std::invalid_argument("Data size mismatch with operator dimensions!");
    }
    
    // Initialize storage vectors
    model.assign(model_size, 0.0f);
    gradient.assign(model_size, 0.0f);
    residual.assign(data_size, 0.0f);
    predicted_data.assign(data_size, 0.0f);
    matvec_result.assign(data_size, 0.0f);
    rmatvec_result.assign(model_size, 0.0f);
    
    // Initialize step size storage
    prev_model.assign(model_size, 0.0f);
    prev_gradient.assign(model_size, 0.0f);
    model_step.assign(model_size, 0.0f);
    gradient_step.assign(model_size, 0.0f);
    
    // Precompute flattened operators
    precompute_operators();
}

void least_squares_migration::precompute_operators() {
    // Reserve space for efficiency
    L_flat.reserve(rows * cols);
    Lt_flat.reserve(rows * cols);
    Lt.assign(cols, std::vector<float>(rows));
    
    // Flatten L
    for (const auto& row : L) {
        L_flat.insert(L_flat.end(), row.begin(), row.end());
    }
    
    // Compute Transpore
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Lt[j][i] = L[i][j];
        }
    }
    
    // Flatten Transpose
    for (const auto& row : Lt) {
        Lt_flat.insert(Lt_flat.end(), row.begin(), row.end());
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
        prev_model = model;
        prev_gradient = gradient;

        // Update Model
        #pragma omp parallel for
        for (size_t i = 0; i < model.size(); i++) {
            model[i] -= alpha * gradient[i];
        }

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
    for (size_t i = 0; i < model_step.size(); i++) {
        model_step[i] = model[i] - prev_model[i];
    }

    // Compute Gradient Step
    for (size_t i = 0; i < gradient_step.size(); i++) {
        gradient_step[i] = gradient[i] - prev_gradient[i];
    }

    // Compute BB Step Sizes
    float s_dot_y = dot_product(model_step, gradient_step);
    float s_dot_s = dot_product(model_step, model_step);
    float y_dot_y = dot_product(gradient_step, gradient_step);

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
    std::vector<float> conj_dir(gradient.size(), 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < conj_dir.size(); i++) {
        conj_dir[i] = -gradient[i];
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        iter_info.n_iter += 1;

        // Store Previous Gradient
        prev_gradient = gradient;

        // Line Search for Optimal Step Size
        alpha = armijo_line_search(conj_dir, 1e-6f);

        // Update Model
        #pragma omp parallel for
        for (size_t i = 0; i < model.size(); i++) {
            model[i] += alpha * conj_dir[i];
        }

        // Compute New Gradient
        compute_gradient();

        // Fletcher-Reeves Formula
        r_norm = dot_product(gradient, gradient);
        r_prev_norm = dot_product(prev_gradient, prev_gradient);

        // Zero Division Check
        if (r_prev_norm < 1e-15f) {
            beta = 0.0f;
        } else {
            beta = r_norm / r_prev_norm;
        }

        // Update Conjugate Direction
        #pragma omp parallel for
        for (size_t i = 0; i < conj_dir.size(); i++) {
            conj_dir[i] = -gradient[i] + beta * conj_dir[i];
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
    float directional_derivative = dot_product(gradient, direction);
    
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
    std::vector<float> original_model = model;
    
    // Take Step
    #pragma omp parallel for
    for (size_t i = 0; i < model.size(); i++) {
        model[i] += step_size * direction[i];
    }
    
    // Evaluate Misfit
    float objective = compute_misfit();
    
    // Restore Original
    model = original_model;
    
    return objective;
}

void least_squares_migration::compute_gradient(){
    // Forward
    start = std::chrono::high_resolution_clock::now();
    linalg::matvec(L_flat, model, rows, cols, predicted_data);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_matvec += 1;
    
    // Compute residual
    compute_residual();
    
    // Adjoint
    start = std::chrono::high_resolution_clock::now();
    linalg::matvec(Lt_flat, residual, cols, rows, gradient);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_rmatvec += 1;
}

void least_squares_migration::compute_residual(){
    #pragma omp parallel for
    for (int i = 0; i < residual.size(); i++) {
        residual[i] = predicted_data[i] - d[i];
    }
}

float least_squares_migration::compute_misfit() {
    float misfit = 0.0f;
    compute_residual();
    
    #pragma omp parallel for reduction(+:misfit)
    for (size_t i = 0; i < residual.size(); i++) {
        misfit += residual[i] * residual[i];
    }
    
    return 0.5f * misfit;
}

float least_squares_migration::dot_product(const std::vector<float>& a, 
                                          const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector size mismatch!");
    }
    float result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}