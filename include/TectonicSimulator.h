#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"
#include <vector>

namespace Ravis {

class TectonicSimulator {
public:
    TectonicSimulator(GoldbergPolyhedron& planet);

    // Inicializa placas mediante Voronoi esférico con domain-warping dependiente de la semilla,
    // asigna tipos de corteza, edades iniciales y espesores isostáticos.
    void generatePlates(int numPlates, const SimulationParameters& params);

    // Ejecuta la simulación tectónica acoplando advección, enfriamiento cortical,
    // orogénesis asimétrica, cadenas de puntos calientes y geomorfología.
    void simulate(int iterations, const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    struct Hotspot {
        Vector3 position;
        float strength = 100.0f;
        float radius = 0.1f;
        bool is_deep = false;
    };
    std::vector<Hotspot> hotspots;

    struct PlateData {
        bool is_oceanic = false;
        Vector3 rotation_axis = Vector3(0, 0, 1);
        float angular_speed = 0.0f;
        Vector3 center = Vector3(1, 0, 0);
    };
    std::vector<PlateData> plates;

    void generateHotspots(const SimulationParameters& params);
    void applyTerrainFeatures(const SimulationParameters& params);
};

} // namespace Ravis