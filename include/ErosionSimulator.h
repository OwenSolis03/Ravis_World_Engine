#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class ErosionSimulator {
public:
    ErosionSimulator(GoldbergPolyhedron& planet);

    // Runs the hydraulic erosion droplet algorithm
    void simulateErosion(int numDrops, const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;
};

} // namespace Ravis
