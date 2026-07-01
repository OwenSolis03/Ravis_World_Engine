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
        float radius = 0.1f;
        bool is_deep = false;
    };
    std::vector<Hotspot> hotspots;

    struct PlateData {
        bool is_oceanic;
        Vector3 rotation_axis;
        float angular_speed;
        Vector3 center; // Seed centroid position
    };
    std::vector<PlateData> plates;

    void simulateStep(const SimulationParameters& params);
    void simulateHotspots();
    void generateHotspots(const SimulationParameters& params);
    void applyTerrainFeatures(const SimulationParameters& params);
};

} // namespace Ravis
