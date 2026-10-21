#pragma once

#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class VegetationSimulator {
public:
    VegetationSimulator(GoldbergPolyhedron& planet);

    // Simula la cobertura vegetal (0..1) a partir del clima, suelo, pendiente y agua ribereña,
    // y aplica retroalimentación biótica (retención de humedad, transpiración, regulación térmica
    // y enriquecimiento orgánico del suelo).
    void simulate(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    void computeCanopyCover(const SimulationParameters& params);
    void applyBioticFeedback(const SimulationParameters& params);
};

} // namespace Ravis