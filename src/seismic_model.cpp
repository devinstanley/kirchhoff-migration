#include "seismic_model.h"
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>

seismic_model::seismic_model(
    std::vector<int> src_coords,
	std::vector<int> rcv_coords,
	int n_ts,
	int n_xs,
	int n_zs,
	double dt,
	double dx,
	double dz,
	double rf,
	double vel):
    src_coords(src_coords),
	rcv_coords(rcv_coords),
	n_ts(n_ts),
	n_xs(n_xs),
	n_zs(n_zs),
	dt(dt),
	dx(dx),
	dz(dz),
	rf(rf),
	vel(vel)
{
    n_srcs = src_coords.size();
	n_rcvs = rcv_coords.size();
}

void seismic_model::generate_model(std::vector<std::vector<float>> points, std::vector<float> amplitudes, float noise) {
	ref_space.resize(n_xs, std::vector<float>(n_zs, 0.0f));

	//Setup Random Distribution
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);


	auto point_to_cord = [this](std::vector<std::vector<float>>& ref_space, std::vector<float> point) {
		int x = point[0] / dx;
		int z = point[1] / dz;

		if (x >= n_xs || z >= n_zs)
			throw std::invalid_argument("Point Outside of Environment");

		return std::make_pair(x, z);
		};

	if (points[0].size() > 1) {
		if (amplitudes.size() != points.size())
			throw std::invalid_argument("Points and amplitudes mismatch");

		for (size_t i = 0; i < points.size(); ++i) {
			auto cord = point_to_cord(ref_space, points[i]);
			ref_space[cord.first][cord.second] = amplitudes[i];
		}
	}
	else {
		auto cord = point_to_cord(ref_space, points[0]);
		ref_space[cord.first][cord.second] = amplitudes[0];
	}
	m.clear();
	for (const auto& row : ref_space)
		m.insert(m.end(), row.begin(), row.end());
	for (auto& val : m)
		val += (distribution(generator) * noise);
}

double seismic_model::ricker_wavelet(double tt) {
	double PI = atan(1.0) * 4;
	double A = powf(PI, 0.25) / sqrtf(2.0 * rf);
	double intermed = pow(PI, 2.0) * pow(rf, 2.0) * pow(tt, 2.0);
	double wavelet = A * (1.0 - 2.0 * intermed) * exp(-1.0 * intermed);
	return wavelet;
}

void seismic_model::m_to_file(const std::string& filename) {
	std::ofstream out(filename);
	out << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::endl;
	if (out.is_open()) {
		for (int ii = 0; ii < m.size(); ii++) {
			out << m[ii] << std::endl;
		}
	}
	out.close();
}