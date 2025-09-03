#include "plot_util.h"

void plot_util::plot_line(const std::vector<float>& y, const std::string& label){
    plt::plot(y);
    plt::title(label);
}

void plot_util::plot_line(const std::vector<float>& x, const std::vector<float>& y, const std::string& label){
    plt::plot(x, y);
    plt::title(label);
}

// Plot Images
void plot_util::plot_image(const std::vector<std::vector<float>>& data, const std::string& label){
    plt::image(data, true);
    plt::colorbar();
    plt::title(label);
}
void plot_util::plot_image(const std::vector<float>& data, const int height, const int width, const std::string& label){
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

// Utilities
void plot_util::create_figure(int height, int width){
    auto handle = plt::figure();
    handle -> size(width, height);
}
void plot_util::subplot(int nrows, int ncols, int index){
    plt::subplot(nrows, ncols, index);
}
void plot_util::show(bool block){
    plt::show();
}
void plot_util::save(const std::string& filename){
    plt::save(filename);
}