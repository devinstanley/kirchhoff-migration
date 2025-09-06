#pragma once

#include "seismic_model.h"

class environment_presets{
    public: 
        enum class presets{
            SINGLE_POINT,
            LINE,
            LAYERS,
            FAULT
        };

        static seismic_model generate_environment(presets preset, int n_ts, int n_xzs, float dt, float dxz, float noise=0, float rf=20, float vel=2500);
};