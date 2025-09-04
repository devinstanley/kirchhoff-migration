#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"
#include "plot_util.h"

int main(int, char**){
    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::LAYERS,
        75,
        60,
        0.002,
        30
    );
    std::cout << "Model Generated" << std::endl;
    plot_util::create_figure(500, 1500);
    plot_util::subplot(1, 3, 0);
    plot_util::plot_image(env.ref_space, "Generated Environment");


    forward_kirchhoff forward(env);
    forward.run();
    std::cout << "Forward Run" << std::endl;

    plot_util::subplot(1, 3, 1);
    plot_util::plot_line(forward.L[0], "L");

    adjoint_kirchhoff adjoint(env);
    adjoint.run(forward.d);
    std::cout << "Adjoint Run" << std::endl;
    plot_util::subplot(1, 3, 2);
    plot_util::plot_image(adjoint.mig, env.n_xs, env.n_zs, "Basic Seismic Migration");
    plt::show();

    return 0;
}
