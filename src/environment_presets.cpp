#include "environment_presets.h"

seismic_model environment_presets::generate_environment(environment_presets::presets preset, int n_src_rcv, int n_ts, int n_xzs, float dt, float dxz, float noise, float rf, float vel){
    std::vector<int> src_cords;
    std::vector<int> rcv_cords;

    float total_extent = n_xzs * dxz;
    
    // Evenly Space
    if (n_src_rcv == 1) {
        // Single At Center
        src_cords.push_back(total_extent / 2.0f);
        rcv_cords.push_back(total_extent / 2.0f);
    } else {
        // Multiple sources/receivers evenly spaced
        float spacing = total_extent / (n_src_rcv - 1);
        for (int ii = 0; ii < n_src_rcv; ii++){
            int position = ii * spacing;
            src_cords.push_back(position);
            rcv_cords.push_back(position);
        }
    }

    seismic_model env(src_cords, rcv_cords, n_ts, n_xzs, n_xzs, dt, dxz, dxz, rf, vel);


    std::vector<std::vector<float>> coords;
    std::vector<float> amps;

    switch (preset)
    {
    case environment_presets::presets::SINGLE_POINT:
        coords = {
            {(n_xzs * dxz) / 2.0f, (n_xzs * dxz) / 2.0f}
        };
        amps = { 1 };
        break;

    case environment_presets::presets::LAYERS:
        for (int ii = 0; ii < n_xzs; ii++){
            // First Layer
            coords.push_back({(n_xzs * dxz) / 2.0f, (float)dxz * ii});
            amps.push_back({ 1 });
            // Second Layer
            coords.push_back({(n_xzs * dxz) / 1.5f, (float)dxz * ii});
            amps.push_back({ 2 });
        }
        break;
    
    case environment_presets::presets::FAULT:
        coords.push_back({15, 15});
        coords.push_back({135, 135});
        coords.push_back({135, 15});
        coords.push_back({75, 75});
        amps.push_back({ 10 });
        amps.push_back({ 50 });
        amps.push_back({ 20 });
        amps.push_back({ 30 });
        break;

    default:
        break;
    }

    env.generate_model(coords, amps, noise);

    return env;
}