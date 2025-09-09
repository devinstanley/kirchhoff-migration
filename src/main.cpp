#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"
#include "plot_util.h"
#include "least_squares.h"
#include <chrono>

int main(int, char**){

    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::LAYERS,
        10, // # of Sources/Receivers
        100, // # of Time Steps
        75, // # of Spatial Steps
        0.002, // Time Step (s)
        2, // Spatial Step (m)
        0,
        20,
        1000
    );

    std::cout << "Model Generated\n";
    plot_util::create_figure(400, 1900);
    plot_util::subplot(1, 4, 0);
    plot_util::plot_image(env.ref_space, "Generated Environment");

    // Generate Synth Data
    forward_kirchhoff forward(env);
    auto start = std::chrono::high_resolution_clock::now();
    forward.run();
    auto end = std::chrono::high_resolution_clock::now();

    // Print Time
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Forward Run: " << elapsed.count() << " seconds\n";

    // Plot
    plot_util::subplot(1, 4, 1);
    plot_util::plot_line(forward.L[0], "L");


    // Standard Mig
    adjoint_kirchhoff adjoint(env);

    start = std::chrono::high_resolution_clock::now();
    adjoint.run(forward.d);
    end = std::chrono::high_resolution_clock::now();

    // Print Time
    elapsed = end - start;
    std::cout << "Adjoint Run: " << elapsed.count() << " seconds\n";

    // Plot Standard
    plot_util::subplot(1, 4, 2);
    plot_util::plot_image(adjoint.mig, env.n_xs, env.n_zs, "Basic Seismic Migration");


    // LSM
    least_squares_migration lsm(env);
    start = std::chrono::high_resolution_clock::now();
    lsm.run(forward.d);
    end = std::chrono::high_resolution_clock::now();

    // Print Time
    elapsed = end - start;
    std::cout << "LSM Run: " << elapsed.count() << " seconds\n";

    // Plot Standard
    plot_util::subplot(1, 4, 3);
    plot_util::plot_image(lsm.get_model(), env.n_xs, env.n_zs, "Least Squares Migration");
    plt::show();

    return 0;
}
