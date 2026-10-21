#pragma once
#include "GoldbergPolyhedron.h"
#include "SimulationParameters.h"

namespace Ravis {

class AtmosphereSimulator {
public:
    AtmosphereSimulator(GoldbergPolyhedron& planet);

    // Fase 1: Temperatura radiativo-convectiva base, vientos preliminares y
    // humedad inicial para alimentar el paso de erosión y corrientes oceánicas.
    void calculatePrimaryClimate(const SimulationParameters& params);

    // Fase 2: Reajuste térmico con anomalías oceánicas (SST), solver SWE de vientos,
    // advección/condensación de humedad en GPU (Clausius-Clapeyron) y biomas.
    void calculateFullClimate(int moistureIterations, const SimulationParameters& params);

    // Clasifica los biomas de Holdridge/Whittaker (se vuelve a llamar tras hidrología).
    void assignBiomes(const SimulationParameters& params);

private:
    GoldbergPolyhedron& planet;

    void calculateTemperatures(const SimulationParameters& params);
    void calculateWinds(const SimulationParameters& params);
    void simulateMoisture(int iterations, const SimulationParameters& params);
};

} // namespace Ravis