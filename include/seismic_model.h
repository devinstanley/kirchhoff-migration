#pragma once

#include <vector>
#include <string>

class seismic_model{
    public:
    //Receivers and Sources
	std::vector<int> src_coords;
	int n_srcs;
	std::vector<int> rcv_coords;
	int n_rcvs;

	// Discretizations
	// Number of Cells
	int n_ts, n_xs, n_zs;

	// Size of Cells
	float dt, dx, dz;

	// Wavelet
	float rf, vel;

	// Containers
	std::vector<std::vector<float>> ref_space;
	std::vector<float> m;

    // Constructor
	seismic_model(
		std::vector<int> src_coords,
		std::vector<int> rcv_coords,
		int n_ts,
		int n_xs,
		int n_zs,
		double dt,
		double dx,
		double dz,
		double rf,
		double vel);

    // Seismic Model Generator
	void generate_model(std::vector<std::vector<float>> points, std::vector<float> amplitudes, float noise = 0);

    // Wavelet Generator
	double ricker_wavelet(double tt);

    // IO Utility
	void save_m_to_file(std::string filename);
};