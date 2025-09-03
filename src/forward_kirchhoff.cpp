#include "forward_kirchhoff.h"

#include <fstream>
#include <iomanip>

forward_kirchhoff::forward_kirchhoff(seismic_model env):env(env){};

void forward_kirchhoff::run() {
	d.assign(env.n_srcs*env.n_rcvs*env.n_ts, 0.0);
	L.assign(env.n_srcs * env.n_rcvs * env.n_ts, std::vector<double>(env.n_zs * env.n_xs, 0.0));


	for (int i_src = 0; i_src < env.n_srcs; i_src++) {
		// Get Source X Pos
		int src_coord = env.src_coords[i_src];

		for (int i_rcv = 0; i_rcv < env.n_rcvs; i_rcv++) {
			// Get Receiver X Pos
			int rcv_coord = env.rcv_coords[i_rcv];

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

					if (env.m[p] > .2){

						for (int it = 0; it < env.n_ts; it++) {
							int u = i_src * env.n_rcvs * env.n_ts + i_rcv * env.n_ts + it;
							double tt = (env.dt * it) - tau_src - tau_rcv;
							double w = env.ricker_wavelet(tt);
							double s = w * env.m[p];
							d[u] = (d[u] + s);
							L[u][p] = w;
						}
					}
				}
			}
		}
	}
}

void forward_kirchhoff::d_to_file() {

	std::ofstream out("ForwardKirchoff_Layer_D.csv");
	out << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::endl;
	if (out.is_open()) {
		for (int ii = 0; ii < d.size(); ii++) {
			out << d[ii] << std::endl;
		}
	}
	out.close();
}

void forward_kirchhoff::L_to_file() {
	
	std::ofstream out("ForwardKirchoff_Layer_L.csv");
	if (out.is_open()) {
		out << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::endl;
		for (std::vector<double> row: L) {
			for (double val : row) {
				out << val << ", ";
			}
			out << std::endl;
		}
		out.close();
	}
}

