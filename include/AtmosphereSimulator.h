#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class AtmosphereSimulator {
public:
    AtmosphereSimulator(GoldbergPolyhedron& planet);

    // Phase 1: Basic temperature and static moisture for erosion
    void calculatePrimaryClimate(const SimulationParameters& params);

    // Phase 2: Wind simulation, dynamic moisture advection, and biome assignment
    void calculateFullClimate(int moistureIterations, const SimulationParameters& params);

    // Re-assign biomes (can be called after external moisture changes)
    void assignBiomes(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    void calculateTemperatures(const SimulationParameters& params);
    void calculateWinds(const SimulationParameters& params);
    void simulateMoisture(int iterations, const SimulationParameters& params);
};

} // namespace Ravis
