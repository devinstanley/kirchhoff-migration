#pragma once

#include <vector>
#include <string>
#include <matplot/matplot.h>

namespace plt = matplot;

class plot_util{
    public:
        // Plot Lines 
        static void plot_line(const std::vector<float>& y, const std::string& label = "");
        static void plot_line(const std::vector<float>& x, const std::vector<float>& y, const std::string& label = "");

        // Plot Images
        static void plot_image(const std::vector<std::vector<float>>& data);
        static void plot_image(const std::vector<float>& data, const int height, const int width);

        // Utilities
        static void create_figure(int height, int width);
        static void subplot(int nrows, int ncols, int index);
        static void show(bool block = false);
        static void save(const std::string& filename);
};