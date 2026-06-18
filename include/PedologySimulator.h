#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class PedologySimulator {
public:
    PedologySimulator(GoldbergPolyhedron& planet);

    // Calculates the surface soil type based on bedrock, climate, and biome
    void generateSoils(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;
};

} // namespace Ravis
