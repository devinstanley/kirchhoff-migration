#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"
#include "environment_presets.h"

int main(int, char**){
    seismic_model env = environment_presets::generate_environment(
        environment_presets::presets::LAYERS,
        150,
        15,
        0.002,
        10
    );
    std::cout << "Model Generated" << std::endl;

    forward_kirchhoff forward(env);
    forward.run();
    std::cout << "Forward Run" << std::endl;

    adjoint_kirchhoff adjoint(env);
    adjoint.run(forward.d);
    std::cout << "Adjoint Run" << std::endl;

    env.m_to_file("m.csv");
    forward.d_to_file("d.csv");
    forward.L_to_file("L.csv");
    adjoint.mig_to_file("mig.csv");
}
