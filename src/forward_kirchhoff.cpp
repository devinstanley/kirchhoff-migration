#include "forward_kirchhoff.h"
#include <omp.h>

#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

forward_kirchhoff::forward_kirchhoff(seismic_model env):env(env){};

void forward_kirchhoff::run() {
	d.assign(env.n_srcs*env.n_rcvs*env.n_ts, 0.0);
	L.assign(env.n_srcs * env.n_rcvs * env.n_ts, std::vector<float>(env.n_zs * env.n_xs, 0.0));

	int n_srcs = static_cast<int>(env.n_srcs);
	int n_rcvs = static_cast<int>(env.n_rcvs);
	int n_xs = static_cast<int>(env.n_xs);
	int n_zs = static_cast<int>(env.n_zs);

	#pragma omp parallel for collapse(4) schedule(static)
	for (int i_src = 0; i_src < n_srcs; i_src++) {
		for (int i_rcv = 0; i_rcv < n_rcvs; i_rcv++) {
			for (int iz = 0; iz < n_zs; iz++) {
				for (int ix = 0; ix < n_xs; ix++) {
					// Get Source X Pos
					int src_coord = env.src_coords[i_src];
					// Get Receiver X Pos
					int rcv_coord = env.rcv_coords[i_rcv];
					//Get Trial X Point
					float x_coord = ix * env.dx;
					//Get Trial Z Point
					float z_coord = iz * env.dz;

					int p = iz * env.n_xs + ix;

					//Calculate Travel Times
					float tau_src = sqrt(z_coord*z_coord + (x_coord - src_coord)*(x_coord - src_coord)) / env.vel;
					float tau_rcv = sqrt(z_coord*z_coord + (x_coord - rcv_coord)*(x_coord - rcv_coord)) / env.vel;

					if (std::abs(env.m[p]) > 1e-6 || true){
						#pragma omp simd
						for (int it = 0; it < env.n_ts; it++) {
							int u = i_src * env.n_rcvs * env.n_ts + i_rcv * env.n_ts + it;
							float tt = (env.dt * it) - tau_src - tau_rcv;
							float w = env.ricker_wavelet(tt);
							float s = w * env.m[p];
							d[u] = (d[u] + s);
							L[u][p] = w;
						}
					}
				}
			}
		}
	}
}

void forward_kirchhoff::d_to_file(const std::string& path) {

	std::ofstream out(path);
	out << std::fixed << std::setprecision(std::numeric_limits<float>::digits10 + 1) << std::endl;
	if (out.is_open()) {
		for (int ii = 0; ii < d.size(); ii++) {
			out << d[ii] << std::endl;
		}
	}
	out.close();
}

void forward_kirchhoff::L_to_file(const std::string& path) {
	
	std::ofstream out(path);
	if (out.is_open()) {
		out << std::fixed << std::setprecision(std::numeric_limits<float>::digits10 + 1) << std::endl;
		for (std::vector<float> row: L) {
			for (float val : row) {
				out << val << ", ";
			}
			out << std::endl;
		}
		out.close();
	}
}

