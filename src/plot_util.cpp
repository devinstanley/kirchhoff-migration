#include "plot_util.h"

void plot_util::plot_line(const std::vector<float>& y, const std::string& label){
    if (!label.empty()) {
        plt::plot(y, {{"label", label}});
        plt::legend();
    } else {
        plt::plot(y);
    }
}

void plot_util::plot_line(const std::vector<float>& x, const std::vector<float>& y, const std::string& label){
    if (!label.empty()) {
        plt::plot(x, y, {{"label", label}});
        plt::legend();
    } else {
        plt::plot(x, y);
    }
}

void plot_util::plot_image(const std::vector<std::vector<float>>& data){
    if (data.empty() || data[0].empty()) {
        throw std::runtime_error("plotImage: empty 2D data");
    }

    int height = static_cast<int>(data.size());
    int width  = static_cast<int>(data[0].size());

    // Flatten
    std::vector<float> buff;
    buff.reserve(height * width);
    for (int i = 0; i < height; i++) {
        if (data[i].size() != static_cast<size_t>(width)) {
            throw std::runtime_error("plotImage: inconsistent row sizes in 2D data");
        }
        buff.insert(buff.end(), data[i].begin(), data[i].end());
    }

    const float* zptr = buff.data();
    plt::imshow(zptr, height, width, 1);
}

void plot_util::plot_image(const std::vector<float>& data, const int height, const int width){
    if (height * width != static_cast<int>(data.size())) {
        throw std::runtime_error("plotImage: size mismatch (height*width != data.size())");
    }
    const float* zptr = data.data();

    plt::imshow(zptr, height, width, 1);
}

void plot_util::subplot(int nrows, int ncols, int index){
    try {
        plt::subplot(nrows, ncols, index);
    } catch (const std::exception& e) {
        std::cerr << "subplot failed: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "subplot failed with unknown error" << std::endl;
        throw;
    }
}

void plot_util::create_figure(int height, int width){
    plt::figure();
    plt::figure_size(width, height);
}

void plot_util::show(bool block){
    plt::show(block);
}

void plot_util::save(const std::string& filename){
    plt::save(filename);
}