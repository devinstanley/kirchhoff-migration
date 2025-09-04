#pragma once

#include <vector>
#include <string>
#include <matplot/matplot.h>

namespace plt = matplot;

class plot_util{
    public:
        // Plot Lines 
        template<typename T>
        static void plot_line(const std::vector<T>& y, const std::string& label = "");

        template<typename T>
        static void plot_line(const std::vector<T>& x, const std::vector<T>& y, const std::string& label = "");

        // Plot Images
        template<typename T>
        static void plot_image(const std::vector<std::vector<T>>& data, const std::string& label = "");

        template<typename T>
        static void plot_image(const std::vector<T>& data, const int height, const int width, const std::string& label = "");

        // Utilities
        static void create_figure(int height, int width);
        static void subplot(int nrows, int ncols, int index);
        static void show(bool block = false);
        static void save(const std::string& filename);
};

template<typename T>
void plot_util::plot_line(const std::vector<T>& y, const std::string& label){
    std::vector<double> y_double(y.begin(), y.end());
    plt::plot(y_double);
    if (!label.empty()) {
        plt::title(label);
    }
}

template<typename T>
void plot_util::plot_line(const std::vector<T>& x, const std::vector<T>& y, const std::string& label){
    std::vector<double> x_double(x.begin(), x.end());
    std::vector<double> y_double(y.begin(), y.end());
    plt::plot(x_double, y_double);
    if (!label.empty()) {
        plt::title(label);
    }
}

// Plot Images
template<typename T>
void plot_util::plot_image(const std::vector<std::vector<T>>& data, const std::string& label){
    std::vector<std::vector<double>> data_double;
    data_double.reserve(data.size());
    
    for (const auto& row : data) {
        data_double.emplace_back(row.begin(), row.end());
    }

    plt::image(data_double, true);
    plt::colorbar();
    plt::title(label);
}

template<typename T>
void plot_util::plot_image(const std::vector<T>& data, const int height, const int width, const std::string& label){
    if (data.size() != static_cast<size_t>(height * width)) {
        throw std::runtime_error("Data size does not match given height*width");
    }

    // Reshape 1D
    std::vector<std::vector<double>> mapped_data(height, std::vector<double>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mapped_data[i][j] = static_cast<double>(data[i * width + j]);
        }
    }

    plt::image(mapped_data, true);
    plt::colorbar();
    plt::title(label);
}