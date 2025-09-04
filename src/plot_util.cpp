#include "plot_util.h"

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