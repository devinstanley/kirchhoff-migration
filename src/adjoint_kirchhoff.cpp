#include "adjoint_kirchhoff.h"

#include <fstream>
#include <iostream>
#include <cmath>

adjoint_kirchhoff::adjoint_kirchhoff(seismic_model env) : env(env) {};

void adjoint_kirchhoff::run(std::vector<float> d) {
	mig.assign(env.n_zs * env.n_xs, 0.0);

	#pragma omp parallel for schedule(static) collapse(3)
	for (int i_src = 0; i_src < env.n_srcs; i_src++) {
		// Get Source X Pos
		int src_coord = env.src_coords[i_src];

		for (int i_rcv = 0; i_rcv < env.n_rcvs; i_rcv++) {
			// Get Receiver X Pos
			int rcv_coord = env.rcv_coords[i_rcv];
			for (int it = 0; it < env.n_ts; it++) {

				int u = i_src * env.n_rcvs * env.n_ts + i_rcv * env.n_ts + it;
				
				for (int iz = 0; iz < env.n_zs; iz++) {
					//Get Trial Z Point
					float z_coord = iz * env.dz;

					#pragma omp simd
					for (int ix = 0; ix < env.n_xs; ix++) {
						//Get Trial X Point
						float x_coord = ix * env.dx;
						int p = iz * env.n_xs + ix;

						//Calculate Travel Times
						float tau_src = sqrtf(z_coord*z_coord + (x_coord - src_coord)*(x_coord - src_coord)) / env.vel;
						float tau_rcv = sqrtf(z_coord*z_coord + (x_coord - rcv_coord)*(x_coord - rcv_coord)) / env.vel;
						float tt = (env.dt * it) - tau_src - tau_rcv;
						float w = env.ricker_wavelet(tt);
						float s = w * d[u];
						
						#pragma omp atomic
						mig[p] += s;
					}
				}
			}
		}
	}
}
void adjoint_kirchhoff::mig_to_file(const std::string& filename) {
	std::ofstream out(filename);
	if (out.is_open()) {
		for (int ii = 0; ii < mig.size(); ii++) {
			out << mig[ii] << std::endl;
		}
	}
	out.close();
}