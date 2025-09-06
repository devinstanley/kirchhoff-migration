#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"
#include "plot_util.h"
#include <chrono>

int main(int, char**){

    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::LAYERS,
        10, // # of Sources/Receivers
        100, // # of Time Steps
        150, // # of Spatial Steps
        0.002, // Time Step (s)
        1, // Spatial Step (m)
        0,
        20,
        1000
    );

    std::cout << "Model Generated\n";
    plot_util::create_figure(500, 1500);
    plot_util::subplot(1, 3, 0);
    plot_util::plot_image(env.ref_space, "Generated Environment");


    forward_kirchhoff forward(env);
    auto start = std::chrono::high_resolution_clock::now();
    forward.run();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Forward Run: " << elapsed.count() << " seconds\n";

    plot_util::subplot(1, 3, 1);
    plot_util::plot_line(forward.L[0], "L");

    adjoint_kirchhoff adjoint(env);

    start = std::chrono::high_resolution_clock::now();
    adjoint.run(forward.d);
    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;
    std::cout << "Adjoint Run: " << elapsed.count() << " seconds\n";

    plot_util::subplot(1, 3, 2);
    plot_util::plot_image(adjoint.mig, env.n_xs, env.n_zs, "Basic Seismic Migration");
    plt::show();

    return 0;
}
