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

        static seismic_model generate_environment(presets preset, int n_ts, int n_xzs, int dt, int dxz, double noise=0, double rf=20, double vel=1000);
};