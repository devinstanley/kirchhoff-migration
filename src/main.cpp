#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"
#include "plot_util.h"
#include "least_squares.h"
#include <chrono>
#include <spgl1_bpdn.h>

int main(int, char**){

    bool do_plot = true;
    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::LAYERS,
        26, // # of Sources/Receivers
        100, // # of Time Steps
        50, // # of Spatial Steps
        0.002, // Time Step (s)
        3, // Spatial Step (m)
        0,
        20,
        1000
    );

    // Generate Synth Data
    forward_kirchhoff forward(env);
    auto start = std::chrono::high_resolution_clock::now();
    forward.run();
    auto end = std::chrono::high_resolution_clock::now();

    // Print Time
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Forward Run: " << elapsed.count() << " seconds\n";


    // Standard Mig
    adjoint_kirchhoff adjoint(env);

    start = std::chrono::high_resolution_clock::now();
    adjoint.run(forward.d);
    end = std::chrono::high_resolution_clock::now();

    // Print Time
    elapsed = end - start;
    std::cout << "Adjoint Run: " << elapsed.count() << " seconds\n";


    // LSM
    least_squares_migration lsm(forward.L, forward.d);
    start = std::chrono::high_resolution_clock::now();
    lsm.run();
    end = std::chrono::high_resolution_clock::now();

    // Print Time
    elapsed = end - start;
    std::cout << "LSM Run: " << elapsed.count() << " seconds\n";

    // SPGL1
    spgl1_bpdn spgl1(forward.L, forward.d, 0.001f);
    start = std::chrono::high_resolution_clock::now();
    spgl1.run(200);
    end = std::chrono::high_resolution_clock::now();

    // Print Time
    elapsed = end - start;
    std::cout << "SPGL1 Run: " << elapsed.count() << " seconds\n";

    // Plot Everything
    if (do_plot){
        plot_util::create_figure(1080, 1920);
        plot_util::subplot(2, 2, 0);
        plot_util::plot_image(env.ref_space, "Generated Environment");
        plot_util::subplot(2, 2, 1);
        plot_util::plot_image(adjoint.mig, env.n_xs, env.n_zs, "Basic Seismic Migration");
        plot_util::subplot(2, 2, 2);
        plot_util::plot_image(lsm.get_model(), env.n_xs, env.n_zs, "Least Squares Migration");
        plot_util::subplot(2, 2, 3);
        plot_util::plot_image(spgl1.x_out, env.n_xs, env.n_zs, "SPGL1 Migration");
        //plt::save("../ex/Seismic_Layers.png");
        plt::show();
    }

    return 0;
}
