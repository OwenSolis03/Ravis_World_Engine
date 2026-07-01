#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class HydrologySimulator {
public:
    HydrologySimulator(GoldbergPolyhedron& planet);

    // Run the full hydrology simulation: flow accumulation + lake detection + riparian moisture
    void simulate(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    // Step 1: Build the drainage network (each cell points to its lowest neighbor)
    void buildDrainageNetwork(const SimulationParameters& params);

    // Step 2: Accumulate precipitation flow downhill
    void accumulateFlow(const SimulationParameters& params);

    // Step 3: Detect endorheic basins and fill lakes
    void detectLakes(const SimulationParameters& params);

    // Step 4: Rivers boost moisture of adjacent cells
    void applyRiparianEffect(const SimulationParameters& params);
};

} // namespace Ravis
