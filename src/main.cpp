#include <iostream>
#include <vector>
#include "seismic_model.h"
#include "forward_kirchhoff.h"
#include "adjoint_kirchhoff.h"

int main(int, char**){
    float noise = 0;
    std::vector<int> src_cords;
    std::vector<int> rcv_cords;
    for (int ii = 0; ii < 15; ii++) {
        src_cords.push_back(ii * 10);
        rcv_cords.push_back(ii * 10);
    }

    seismic_model env(src_cords, rcv_cords, 150, 15, 15, 0.002, 10, 10, 20.0, 1000);

    std::vector<std::vector<float>> coords
    {
        {74, 74}
    };
    std::vector<float> amps{ 1 };
    env.generate_model(coords, amps, noise);

    forward_kirchhoff forward(env);
    forward.run();

    adjoint_kirchhoff adjoint(env);
    adjoint.run(forward.d);

    env.m_to_file("m.csv");
    forward.d_to_file("d.csv");
    forward.L_to_file("L.csv");
    adjoint.mig_to_file("mig.csv");
}
