#include "spgl1_bpdn.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>

spgl1_bpdn::spgl1_bpdn(
	std::vector<std::vector<float>>& A, 
	std::vector<float>& b, 
	float sigma, params args, linalg_backends backend):
	A(A), b(b), sigma(sigma), args(args)
{
    info iter_info = info();
    rows = A.size();
    cols = A[0].size();
    verbosity = 1;

    ops = linalg_dispatch::get_ops(backend);

    // Reserve Space
    x_out.assign(cols, 0.0f);
    f_vals.assign(10, std::numeric_limits<float>::min());
    x_grad.assign(cols, 0.0f);
    g_grad.assign(rows, 0.0f);
    matvec_result.assign(rows, 0.0f);
    rmatvec_result.assign(cols, 0.0f);

    precompute_operators();
    set_optimal_params();
}

void spgl1_bpdn::precompute_operators(){
    // Precompute Raveled A and A_T
    A_flat.reserve(rows * cols);
    At_flat.reserve(rows * cols);
    At.assign(cols, std::vector<float>(rows));

    for (const auto& row : A) {
		A_flat.insert(A_flat.end(), row.begin(), row.end());
	}
    for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			At[j][i] = A[i][j];
		}
	}
    for (const auto& row : At) {
		At_flat.insert(At_flat.end(), row.begin(), row.end());
	}
}

void spgl1_bpdn::set_optimal_params(){
    sigma = 1e-4 * ops.l2_norm(b);
    ops.matvec(At_flat, b, cols, rows, rmatvec_result);
    tau = 0.1 * ops.inf_norm(ops.scalar_vector_prod(-1.0, rmatvec_result));
    if (verbosity > 0){
        std::cout << "Optimal Sigma:\t" << sigma << "\tOptimal Tau:\t" << tau << std::endl;
    }
}

void spgl1_bpdn::run(int max_iter) {
    if (max_iter <= 0) {
        throw std::invalid_argument("Max iterations must be positive!");
    }

	/*
	Reference
    ----------
    [1] E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
             for basis pursuit solutions", SIAM J. on Scientific Computing,
             31(2):890-912. (2008).
	*/

    // Set Initial Iterates
    bnorm = ops.l2_norm(b);
    auto start = std::chrono::high_resolution_clock::now();
	x_out = ops.l1_norm_projection(x_out, tau);
	auto stop = std::chrono::high_resolution_clock::now();
	iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

	start = std::chrono::high_resolution_clock::now();
	ops.matvec(A_flat, x_out, rows, cols, matvec_result);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	iter_info.n_matvec += 1;

	r = (ops.vector_subtract(b, matvec_result));
	
	start = std::chrono::high_resolution_clock::now();
	ops.rmatvec(At_flat, r, cols, rows, rmatvec_result);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	iter_info.n_rmatvec += 1;

	g = ops.scalar_vector_prod(-1.0, rmatvec_result);

    // Objective Function
	f = pow(ops.l2_norm(r), 2.0) / 2.0;
	f_vals[0] = f;
	f_best = f;
	f_old = f;
	x_best = x_out;

    //Compute Projected Gradient Direction
	start = std::chrono::high_resolution_clock::now();
	dx = ops.vector_subtract(ops.l1_norm_projection(ops.vector_subtract(x_out, g), tau), x_out);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    // Set Initial Step
    dx_norm = ops.inf_norm(dx);
	if (dx_norm < (1 / args.max_step)) {
		g_step = args.max_step;
	}
	else {
		g_step = std::min(args.max_step, std::max(args.min_step, 1.0f / dx_norm));
	}

    // Main While Loop
    while (1) {
        // Compute Logging and Exit Conditions
        compute_exit_conditions();
        if (verbosity == 1) {
			std::printf("%7d\t%8e\t%8e\t%8e\t%8e\t%8e\t%10d\t%10d \n", 
				iter_info.n_iter, rnorm, rgap, rerror1, rerror2, gnorm, iter_info.n_matvec, iter_info.n_rmatvec);
		}

        // Check Exit Conditions
        if (check_exit_conditions()) {
            break;
        }
        if (iter_info.n_iter >= max_iter){
            std::cout << "Max Iterations Reached!" << std::endl;
            break;
        }

        // Update Tau If Needed
        update_tau();

        // Update Iterates
        update_iterates();

        // Projected Gradients Step and Linesearch
        // Try Line Curvy Search
        bool line_err = curve_line_search();
        // Failed - Try w Feasable Dirn Search
        if (line_err){
            // Reset Iterates
            x_out = x_old;
            f = f_old;

            line_err = dirn_line_search();
        }

        // Failed Again -> Revert to Previous Iterate and Damp Max Step
        if (line_err){
            x_out = x_old;
			f = f_old;

            if (1) {
				args.max_step = args.max_step / 10;
				std::cout << "Linesearch Failed, Damping Max BB Scaling to: " << args.max_step << std::endl;
			}
        }

        if (!line_err){
            compute_gradient();
        }
        else{
            g_step = std::min(args.max_step, g_step);
        }

        // Update Function History
		if (f > pow(sigma, 2.0) / 2) {
			f_vals[iter_info.n_iter % 3] = f;
			if (f_best > f) {
				f_best = f;
				x_best = x_out;
			}
		}
    }

    std::cout << "Completed SPGL1 Migration" << std::endl;

    return;
}

void spgl1_bpdn::compute_exit_conditions() {
    gnorm = ops.inf_norm(ops.scalar_vector_prod(-1.0, g));
    rnorm = ops.l2_norm(r);
    gap = ops.dot(r, ops.vector_subtract(r, b)) + tau * gnorm;
    rgap = abs(gap) / std::max(1.0f, f);
    aerror1 = rnorm - sigma;
    aerror2 = f - (sigma * sigma) / 2.0f;
    rerror1 = abs(aerror1) / std::max(1.0f, rnorm);
    rerror2 = abs(aerror2) / std::max(1.00f, f);
}

bool spgl1_bpdn::check_exit_conditions() {
    if (gnorm <= args.ls_tol * rnorm) {
        std::cout << "Exit Least Squares" << std::endl;
        return true;
    }
    if (args.max_step < 1e-12){
        std::cout << "Step Size Collapsed" << std::endl;
        return true;
    }
    if (rgap <= std::max(args.opt_tol, rerror2) or rerror1 <= args.opt_tol) {
        if (rnorm <= sigma) {
            std::cout << "Suboptimal BP Sol" << std::endl;
            return true;
        }
        if (rerror1 <= args.opt_tol) {
            std::cout << "Found Approx Root" << std::endl;
            return true;
        }
        if (rnorm <= args.bp_tol*bnorm) {
            std::cout << "Found BP Solution" << std::endl;
            return true;
        }
    }
    return false;
}

void spgl1_bpdn::update_tau() {
    // Check if Tau Needs to be Updated
    fchange = abs(f - f_old);
    if (fchange <= args.dec_tol * f) {
        relchange1 = true;
    }
    else {
        relchange1 = false;
    }
    if (fchange <= 1e-1 * f * abs(rnorm - sigma)) {
        relchange2 = true;
    }
    else {
        relchange2 = false;
    }
    if ((relchange1 and (rnorm > 2.0 * sigma)) or (relchange2 and (rnorm <= 2.0 * sigma))) {} // Need to Update Tau
    else {
        return;
    }

    // Update Tau
    tau_old = tau;
    tau = std::max(0.0f, tau + ((rnorm * aerror1) / gnorm));
    iter_info.n_newton += 1;

    if (tau < tau_old) {
        start = std::chrono::high_resolution_clock::now();
        x_out = ops.l1_norm_projection(x_out, tau);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        start = std::chrono::high_resolution_clock::now();
        ops.matvec(A_flat, x_out, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = (ops.vector_subtract(b, matvec_result));


        start = std::chrono::high_resolution_clock::now();
        ops.rmatvec(At_flat, r, cols, rows, rmatvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_rmatvec += 1;

        g = ops.scalar_vector_prod(-1.0, rmatvec_result);

        f = pow(ops.l2_norm(r), 2.0) / 2.0;
        f_vals.assign(10, -1000000.0);
        f_vals.push_back(f);
    }
}

void spgl1_bpdn::update_iterates() {
    iter_info.n_iter += 1;
    x_old = x_out;
    f_old = f;
    g_old = g;
    r_old = r;
}

bool spgl1_bpdn::curve_line_search() {
    // Set Parameters
    step = 1;
    snorm = 0;
    scale = 1;
    nsafe = 0;
    n_iters = 0;
    b_search = ops.scalar_vector_prod(g_step, g);

    // Line Search Start
    while (1) {
        start = std::chrono::high_resolution_clock::now();
        x_out = ops.l1_norm_projection(ops.vector_subtract(x_old, ops.scalar_vector_prod(step * scale, b_search)), tau);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        start = std::chrono::high_resolution_clock::now();
        ops.matvec(A_flat, x_out, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = ops.vector_subtract(b, matvec_result);

        f = abs(ops.dot(r, r)) / 2.0f;
        s = ops.vector_subtract(x_out, x_old);
        float gts = scale*ops.dot(b_search, s);

        //Error Exit Conditions
        //Negative Descent
        if (gts >= 0) {
            return true;
        }
        //Max iterations
        if (n_iters >= 10) {
            return true;
        }
        //Exit Condition 
        if (f < *std::max_element(f_vals.begin(), f_vals.end()) + gamma * step * gts) {
            return false;
        }

        //New Iteration Vars
        n_iters += 1;
        step /= 2;

        //Dampen Search
        snormold = snorm;
        snorm = ops.l2_norm(s) / sqrt((float)x_out.size());
        if (abs(snorm - snormold) <= 1e-6 * snorm) {
            gnorm = ops.l2_norm(b_search) / sqrt((double)x_out.size());
            scale = snorm / gnorm / pow(2.0, nsafe);
            nsafe += 1;
        }
    }
}

bool spgl1_bpdn::dirn_line_search() {
    // Set Parameters
    step = 1;
    n_iters = 0;
    start = std::chrono::high_resolution_clock::now();
    dx = ops.vector_subtract(ops.l1_norm_projection(ops.vector_subtract(x_out, ops.scalar_vector_prod(g_step, g)), tau), x_out);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    
    gtd = ops.dot(g, dx);
    if (gtd >= 0) {  // Not a Descent Direction
        return true;
    }

    // Line Search Start
    while (1) {
        x_out = ops.vector_subtract(x_old, ops.scalar_vector_prod(step, dx));

        start = std::chrono::high_resolution_clock::now();
        ops.matvec(A_flat, x_out, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = ops.vector_subtract(b, matvec_result);


        f = abs(ops.dot(r, r)) / 2.0;
        if (f < *std::max_element(f_vals.begin(), f_vals.end()) + gamma * step * gtd) {
            return false;
        }
        if (n_iters >= 10) {
            return true;
        }
        if (step <= 0.1) {
            step /= 2;
        }
        else {
            float denominator = 2 * (f - f_old - step * gtd);
            if (std::abs(denominator) > 1e-10) {  // Avoid division by zero
                tmp = (-gtd * step * step) / denominator;
                if (tmp < 0.1 || tmp > 0.9 * step || !std::isfinite(tmp)) {
                    tmp = step / 2;
                }
                step = tmp;
            } else {
                step /= 2;
            }
        }

        n_iters += 1;
    }
}

void spgl1_bpdn::compute_gradient() {
    start = std::chrono::high_resolution_clock::now();
    ops.rmatvec(At_flat, r, cols, rows, rmatvec_result);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_rmatvec += 1;

    g = ops.scalar_vector_prod(-1.0, rmatvec_result);

    x_grad = ops.vector_subtract(x_out, x_old);
    g_grad = ops.vector_subtract(g, g_old);
    if (ops.dot(x_grad, g_grad) <= 0) {
        g_step = args.max_step;
    }
    else {
        g_step = std::min(args.max_step, std::max(args.min_step, ops.dot(x_grad, x_grad) / ops.dot(x_grad, g_grad)));
    }
}