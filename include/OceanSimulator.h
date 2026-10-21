#pragma once

#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class OceanSimulator {
public:
    OceanSimulator(GoldbergPolyhedron& p);

    // Simula las corrientes oceánicas superficiales (Ekman + giros subtropicales/subpolares de Munk)
    // y calcula el transporte de calor marino (anomalías SST y surgencia costera).
    void simulateCurrents(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;
};

} // namespace Ravis