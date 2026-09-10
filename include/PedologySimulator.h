#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class PedologySimulator {
public:
    PedologySimulator(GoldbergPolyhedron& planet);

    // Refines Cell::bedrock from tectonic setting + climate (mountains ->
    // metamorphic, dry basins -> sandstone, shelves / warm lowlands ->
    // shale-limestone, deep ocean -> basalt, cratons -> granite). Call before
    // generateSoils().
    void classifyBedrock(const SimulationParameters& params);

    // Calculates the surface soil type based on bedrock, climate, and biome
    void generateSoils(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;
};

} // namespace Ravis
