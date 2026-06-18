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

private:
    GoldbergPolyhedron& planet;

    void calculateTemperatures(const SimulationParameters& params);
    void calculateWinds();
    void simulateMoisture(int iterations, const SimulationParameters& params);
    void assignBiomes();
};

} // namespace Ravis
