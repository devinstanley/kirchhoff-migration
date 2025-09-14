#include "spgl1_bpdn.h"
#include "linalg.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>

spgl1_bpdn::spgl1_bpdn(
	std::vector<std::vector<float>>& A, 
	std::vector<float>& b, 
	float sigma, params args):
	A(A), b(b), sigma(sigma), args(args)
{
    info iter_info = info();
    rows = A.size();
    cols = A[0].size();
    verbosity = 1;
}

void spgl1_bpdn::run(int max_iter) {
	/*
	Reference
    ----------
    [1] E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
             for basis pursuit solutions", SIAM J. on Scientific Computing,
             31(2):890-912. (2008).
	*/

    // Reserve Space
    init_x.assign(cols, 0.0f);
    fvals.assign(10, std::numeric_limits<float>::min());
    xgrad.assign(cols, 0.0f);
    ggrad.assign(rows, 0.0f);
    trans.assign(cols, std::vector<float>(rows));
    At_flat.reserve(rows * cols);
    A_flat.reserve(rows * cols);
    matvec_result.assign(rows, 0.0f);
    rmatvec_result.assign(cols, 0.0f);

    // Precompute Raveled A and A_T
    for (const auto& row : A) {
		A_flat.insert(A_flat.end(), row.begin(), row.end());
	}
    for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			trans[j][i] = A[i][j];
		}
	}
    for (const auto& row : trans) {
		At_flat.insert(At_flat.end(), row.begin(), row.end());
	}

    // Set Initial Iterates
    bnorm = linalg::l2_norm(b);
    auto start = std::chrono::high_resolution_clock::now();
	x = linalg::l1_norm_projection(init_x, tau);
	auto stop = std::chrono::high_resolution_clock::now();
	iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

	start = std::chrono::high_resolution_clock::now();
	linalg::matvec(A_flat, x, rows, cols, matvec_result);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	iter_info.n_matvec += 1;

	r = (linalg::vector_subtract(b, matvec_result));
	
	start = std::chrono::high_resolution_clock::now();
	linalg::matvec(At_flat, r, cols, rows, rmatvec_result);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	iter_info.n_rmatvec += 1;

	g = linalg::scalar_vector_prod(-1.0, rmatvec_result);

    // Objective Function
	f = pow(linalg::l2_norm(r), 2.0) / 2.0;
	fvals[0] = f;
	fbest = f;
	fold = f;
	xbest = x;

    //Compute Projected Gradient Direction
	start = std::chrono::high_resolution_clock::now();
	dx = linalg::vector_subtract(linalg::l1_norm_projection(linalg::vector_subtract(x, g), tau), x);
	stop = std::chrono::high_resolution_clock::now();
	iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    // Set Initial Step
    dx_norm = linalg::inf_norm(dx);
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
            x = xold;
            f = fold;

            line_err = dirn_line_search();
        }

        // Failed Again -> Revert to Previous Iterate and Damp Max Step
        if (line_err){
            x = xold;
			f = fold;

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
			fvals[iter_info.n_iter % 3] = f;
			if (fbest > f) {
				fbest = f;
				xbest = x;
			}
		}
    }

    std::cout << "Completed SPGL1 Migration" << std::endl;
    x_out = x;

    return;
}

void spgl1_bpdn::compute_exit_conditions() {
    gnorm = linalg::inf_norm(linalg::scalar_vector_prod(-1.0, g));
    rnorm = linalg::l2_norm(r);
    gap = linalg::dot(r, linalg::vector_subtract(r, b)) + tau * gnorm;
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
    fchange = abs(f - fold);
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
        x = linalg::l1_norm_projection(x, tau);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        start = std::chrono::high_resolution_clock::now();
        linalg::matvec(A_flat, x, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = (linalg::vector_subtract(b, matvec_result));


        start = std::chrono::high_resolution_clock::now();
        linalg::matvec(At_flat, r, cols, rows, rmatvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_rmatvec += 1;

        g = linalg::scalar_vector_prod(-1.0, rmatvec_result);

        f = pow(linalg::l2_norm(r), 2.0) / 2.0;
        fvals.assign(10, -1000000.0);
        fvals.push_back(f);
    }
}

void spgl1_bpdn::update_iterates() {
    iter_info.n_iter += 1;
    xold = x;
    fold = f;
    gold = g;
    rold = r;
}

bool spgl1_bpdn::curve_line_search() {
    // Set Parameters
    step = 1;
    snorm = 0;
    scale = 1;
    nsafe = 0;
    n_iters = 0;
    b_search = linalg::scalar_vector_prod(g_step, g);

    // Line Search Start
    while (1) {
        start = std::chrono::high_resolution_clock::now();
        x = linalg::l1_norm_projection(linalg::vector_subtract(xold, linalg::scalar_vector_prod(step * scale, b_search)), tau);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        start = std::chrono::high_resolution_clock::now();
        linalg::matvec(A_flat, x, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = linalg::vector_subtract(b, matvec_result);

        f = abs(linalg::dot(r, r)) / 2.0f;
        s = linalg::vector_subtract(x, xold);
        float gts = scale*linalg::dot(b_search, s);

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
        if (f < *std::max_element(fvals.begin(), fvals.end()) + gamma * step * gts) {
            return false;
        }

        //New Iteration Vars
        n_iters += 1;
        step /= 2;

        //Dampen Search
        snormold = snorm;
        snorm = linalg::l2_norm(s) / sqrt((float)x.size());
        if (abs(snorm - snormold) <= 1e-6 * snorm) {
            gnorm = linalg::l2_norm(b_search) / sqrt((double)x.size());
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
    dx = linalg::vector_subtract(linalg::l1_norm_projection(linalg::vector_subtract(x, linalg::scalar_vector_prod(g_step, g)), tau), x);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.proj_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    gtd = -1.0 * abs(linalg::dot(g, dx));

    // Line Search Start
    while (1) {
        x = linalg::vector_subtract(x, linalg::scalar_vector_prod(step, dx));

        start = std::chrono::high_resolution_clock::now();
        linalg::matvec(A_flat, x, rows, cols, matvec_result);
        stop = std::chrono::high_resolution_clock::now();
        iter_info.matvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        iter_info.n_matvec += 1;

        r = linalg::vector_subtract(b, matvec_result);


        f = abs(linalg::dot(r, r)) / 2.0;
        if (f < *std::max_element(fvals.begin(), fvals.end()) + gamma * step * gtd) {
            return false;
        }
        if (n_iters >= 10) {
            return true;
        }
        if (step <= 0.1) {
            step /= 2;
        }
        else {
            tmp = (-1 * gtd * pow(step, 2)) / (2 * (f - fold - step * gtd));
            if (tmp < 0.1 or tmp > 0.9 * step) {
                tmp = step / 2;
            }
            step = tmp;
        }

        n_iters += 1;
    }
}

void spgl1_bpdn::compute_gradient() {
    start = std::chrono::high_resolution_clock::now();
    linalg::matvec(At_flat, r, cols, rows, rmatvec_result);
    stop = std::chrono::high_resolution_clock::now();
    iter_info.rmatvec_time += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    iter_info.n_rmatvec += 1;

    g = linalg::scalar_vector_prod(-1.0, rmatvec_result);

    xgrad = linalg::vector_subtract(x, xold);
    ggrad = linalg::vector_subtract(g, gold);
    if (linalg::dot(xgrad, ggrad) <= 0) {
        g_step = args.max_step;
    }
    else {
        g_step = std::min(args.max_step, std::max(args.min_step, linalg::dot(xgrad, xgrad) / linalg::dot(xgrad, ggrad)));
    }
}