#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"
#include "plot_util.h"

int main(int, char**){
    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::SINGLE_POINT,
        150,
        15,
        0,
        10
    );
    std::cout << "Model Generated" << std::endl;
    plot_util::plot_image(env.ref_space);
    plot_util::create_figure(50, 50);
    plot_util::show(true);

    forward_kirchhoff forward(env);
    forward.run();
    std::cout << "Forward Run" << std::endl;

    adjoint_kirchhoff adjoint(env);
    adjoint.run(forward.d);
    std::cout << "Adjoint Run" << std::endl;
    plot_util::plot_image(std::vector<float> (adjoint.mig.begin(), adjoint.mig.end()), env.n_xs, env.n_zs);
    plot_util::show(true);

    //env.m_to_file("m.csv");
    //forward.d_to_file("d.csv");
    //forward.L_to_file("L.csv");
    //adjoint.mig_to_file("mig.csv");
    return 0;
}
