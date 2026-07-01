#pragma once

#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class OceanSimulator {
public:
    OceanSimulator(GoldbergPolyhedron& p);

    // Simulates wind-driven ocean currents solving Munk's stream function
    void simulateCurrents(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;
};

} // namespace Ravis
