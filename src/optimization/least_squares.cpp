#include <iostream>

#include "least_squares.h"
#include <limits>
#include <numeric>


least_squares_migration::least_squares_migration(seismic_model& env): 
    env(env), fwd(env), adj(env), data_ptr(nullptr)
{
    int model_size = env.n_zs * env.n_xs;
    int data_size = env.n_srcs * env.n_rcvs * env.n_ts;

    if (model_size <= 0 || data_size <= 0){
        throw std::invalid_argument("Invalid model dimensions!");
    }

    // Initialize LSM Storage
    model.assign(model_size, 0.0f);
    gradient.assign(model_size, 0.0f);
    residual.assign(data_size, 0.0f);
    predicted_data.assign(data_size, 0.0f);

    // Initialize Step Size Storage
    prev_model.assign(model_size, 0.0f);
    prev_gradient.assign(model_size, 0.0f);
    model_step.assign(model_size, 0.0f);
    gradient_step.assign(model_size, 0.0f);
};

void least_squares_migration::run(const std::vector<float>& data, optimizers optimizer, int max_iterations, float tol, int verbosity){
    // Input Validation
    if (data.empty()) {
        throw std::invalid_argument("Input data is empty!");
    }
    
    int expected_data_size = env.n_srcs * env.n_rcvs * env.n_ts;
    if (data.size() != expected_data_size) {
        throw std::invalid_argument("Input data does not match expected model size!");
    }
    
    if (max_iterations <= 0) {
        throw std::invalid_argument("Max iterations must be positive!");
    }

    this->data_ptr = &data;

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
            if (verbosity > 1 || (verbosity > 0 && iter % 10 == 0)){
                std::cout << "\tBB step size: " << alpha << std::endl;
            }
        }

        // Compute Misfit
        float current_misfit = compute_misfit();
        float misfit_reduction = prev_misfit - current_misfit;

        if (verbosity > 1 || (verbosity > 0 && iter % 10 == 0)){
            std::cout << "Iteration " << iter << ": Misfit = " << current_misfit 
                << ", Reduction = " << misfit_reduction << std::endl;
            std::cout << "\tCurrent model sum: " << std::accumulate(predicted_data.begin(), predicted_data.end(), 0.0f) << std::endl;

            float pred_sum = std::accumulate(fwd.d.begin(), fwd.d.end(), 0.0f);
            std::cout << "\tPredicted data sum: " << pred_sum << std::endl;

            float true_data_sum = std::accumulate(data_ptr->begin(), data_ptr->end(), 0.0f);
            std::cout << "\tTrue data sum: " << true_data_sum << std::endl;
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
    float alpha = DEFAULT_STEP_SIZE;
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
        // Store Previous Gradient
        std::vector<float> prev_gradient = gradient;

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
            std::cout << "CG Iteration " << iter << ": Misfit = " << current_misfit 
                      << ", Reduction = " << misfit_reduction 
                      << ", Beta = " << beta << std::endl;
        }
        
        // Check Convergence
        if (std::abs(misfit_reduction) < tol && iter > 0) {
            std::cout << "Converged after " << iter << " iterations" << std::endl;
            break;
        }

        prev_misfit = current_misfit;
    }
}

void least_squares_migration::compute_gradient(){
    // Calculate New Prediction
    env.m = model;
    fwd.run();

    // Store Prediction
    #pragma omp parallel for
    for (size_t i = 0; i < predicted_data.size(); i++) {
        predicted_data[i] = fwd.d[i];
    }

    // Compute Residual
    compute_residual();

    // Compute Gradient
    adj.run(residual);

    // Store Gradient
    #pragma omp parallel for
    for (size_t i = 0; i < gradient.size(); i++) {
        gradient[i] = adj.mig[i];
    }
}

void least_squares_migration::compute_residual(){
    const std::vector<float>& data = *data_ptr; 
    
    #pragma omp parallel for
    for (size_t i = 0; i < residual.size(); i++) {
        residual[i] = predicted_data[i] - data[i];
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