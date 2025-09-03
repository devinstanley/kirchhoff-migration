#include "adjoint_kirchhoff.h"

#include <fstream>

adjoint_kirchhoff::adjoint_kirchhoff(seismic_model env) : env(env) {};

void adjoint_kirchhoff::run(std::vector<double> d) {
	mig.assign(env.n_zs * env.n_xs, 0.0);

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
					int z_coord = iz * env.dz;

					for (int ix = 0; ix < env.n_xs; ix++) {
						//Get Trial X Point
						int x_coord = ix * env.dx;
						int p = ix * env.n_zs + iz;

						//Calculate Travel Times
						double tau_src = sqrt(pow(z_coord, 2.0) + pow(x_coord - src_coord, 2.0)) / env.vel;
						double tau_rcv = sqrt(pow(z_coord, 2.0) + pow(x_coord - rcv_coord, 2.0)) / env.vel;

						double tt = (env.dt * it) - tau_src - tau_rcv;

						double w = env.ricker_wavelet(tt);
						double s = w * d[u];
						mig[p] = mig[p] + s;
					}
				}
			}
		}
	}
}
void adjoint_kirchhoff::mig_to_file(std::string filename) {
	std::ofstream out(filename);
	if (out.is_open()) {
		for (int ii = 0; ii < mig.size(); ii++) {
			out << mig[ii] << std::endl;
		}
	}
	out.close();
}