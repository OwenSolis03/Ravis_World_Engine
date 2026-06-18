#pragma once
#include "GoldbergPolyhedron.h"
#include <vector>

#include "SimulationParameters.h"

namespace Ravis {

class TectonicSimulator {
public:
    TectonicSimulator(GoldbergPolyhedron& planet);

    // Initializes plates using spherical Voronoi (multi-source BFS)
    // Randomly assigns Oceanic or Continental crust types and base elevations.
    void generatePlates(int numPlates, const SimulationParameters& params);

    // Run the geological simulation for a number of iterations
    void simulate(int iterations, const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    struct Hotspot {
        Vector3 position;
        float strength;
    };
    std::vector<Hotspot> hotspots;

    void simulateStep(const SimulationParameters& params);
    void simulateHotspots();
    void generateHotspots(int numHotspots);
};

} // namespace Ravis
