#include "../include/VegetationSimulator.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace Ravis {

VegetationSimulator::VegetationSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void VegetationSimulator::simulate(const SimulationParameters& params) {
    computeCanopyCover(params);
    applyBioticFeedback(params);
}

void VegetationSimulator::computeCanopyCover(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const int n = static_cast<int>(cells.size());
    if (n == 0) return;

    const float sea = params.effective_sea_level();

    std::vector<float> cover(n, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto& cell = cells[i];
        if (cell.elevation <= sea) {
            cover[i] = 0.0f;
            continue;
        }

        const float tempC = cell.temperature * 55.0f - 15.0f;
        const float elevAbove = cell.elevation - sea;

        // Sin vegetación bajo casquetes polares o cumbres nivales extremas
        if (tempC < -6.0f || elevAbove > 4800.0f) {
            cover[i] = 0.0f;
            continue;
        }

        // 1. Factor térmico fotosintético (óptimo entre 10 °C y 28 °C, decae hacia heladas y calor extremo)
        float thermal_suitability = 0.0f;
        if (tempC < 10.0f) {
            thermal_suitability = (tempC - (-6.0f)) / 16.0f; // 0 en -6°C -> 1.0 en +10°C
        } else if (tempC <= 28.0f) {
            thermal_suitability = 1.0f;
        } else {
            thermal_suitability = std::max(0.35f, 1.0f - (tempC - 28.0f) * 0.035f);
        }

        // 2. Retención de agua del suelo (Pedology coupling)
        float soil_retention = 0.85f;
        switch (cell.soil) {
            case SoilType::LOAM: soil_retention = 1.20f; break; // Suelo franco humífero retiene humedad óptima
            case SoilType::CLAY: soil_retention = 1.05f; break; // Arcilla retiene bien en trópicos
            case SoilType::SAND: soil_retention = 0.55f; break; // Arena filtra rápido (dificulta bosque denso)
            case SoilType::NONE: soil_retention = 0.15f; break; // Roca desnuda
        }

        // 3. Disponibilidad hídrica efectiva (lluvia + humedad freática de ríos/lagos vs demanda evaporativa)
        float t_bio = std::max(1.0f, std::min(30.0f, tempC));
        float pet_norm = (52.0f * t_bio + 120.0f) / 4000.0f; // Demanda evaporativa en escala 0..1
        float riparian_water = cell.is_lake ? 0.25f : std::min(0.22f, std::sqrt(cell.river_flow) * 0.045f);
        float effective_water = (cell.precipitation + 0.35f * cell.moisture + riparian_water) * soil_retention;

        float aridity_ratio = effective_water / std::max(0.05f, pet_norm);
        // Curva logística suave: < 0.18 desierto desnudo, 0.35 pastizal/matorral, > 0.85 bosque cerrado
        float water_suitability = std::max(0.0f, std::min(1.0f, (aridity_ratio - 0.14f) / 0.82f));

        // 4. Penalización por pendiente escarpada (los acantilados no sostienen dosel cerrado)
        float max_diff = 0.0f;
        for (size_t nid : cell.neighbors) {
            float d = std::abs(cells[nid].elevation - cell.elevation);
            if (d > max_diff) max_diff = d;
        }
        float slope_factor = std::max(0.20f, 1.0f - std::max(0.0f, max_diff - 500.0f) / 1800.0f);

        cover[i] = std::max(0.0f, std::min(1.0f, thermal_suitability * water_suitability * slope_factor));
    }

    // 2 pasadas de dispersión ecológica (colonización de semillas y microclima de borde de bosque)
    std::vector<float> next_cover(n, 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (cells[i].elevation <= sea) {
                next_cover[i] = 0.0f;
                continue;
            }
            float sum = 0.0f;
            int cnt = 0;
            for (size_t nid : cells[i].neighbors) {
                if (cells[nid].elevation > sea) {
                    sum += cover[nid];
                    cnt++;
                }
            }
            float nbr_mean = (cnt > 0) ? (sum / static_cast<float>(cnt)) : cover[i];
            next_cover[i] = 0.72f * cover[i] + 0.28f * nbr_mean;
        }
        cover.swap(next_cover);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        cells[i].vegetation_cover = cover[i];
    }
}

void VegetationSimulator::applyBioticFeedback(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const int n = static_cast<int>(cells.size());
    const float sea = params.effective_sea_level();

    // Calcular cuánta humedad transpiran los bosques viento arriba hacia las celdas vecinas
    std::vector<float> transpired_precip(n, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (cells[i].elevation <= sea) continue;

        float v = cells[i].vegetation_cover;
        float nbr_v = 0.0f;
        int land_nbrs = 0;
        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].elevation > sea) {
                nbr_v += cells[nid].vegetation_cover;
                land_nbrs++;
            }
        }
        if (land_nbrs > 0) nbr_v /= static_cast<float>(land_nbrs);

        float regional_canopy = 0.6f * v + 0.4f * nbr_v;

        // 1. Retención y reciclaje de humedad por transpiración ("Bomba Biótica"):
        // Los bosques retienen agua en el suelo y aumentan la lluvia convectiva local hasta un +18%
        transpired_precip[i] = regional_canopy * cells[i].moisture * 0.16f;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        auto& cell = cells[i];
        if (cell.elevation <= sea) continue;

        const float v = cell.vegetation_cover;
        if (v <= 0.01f) continue;

        // Actualizar humedad y precipitación con el reciclaje vegetal
        cell.moisture = std::min(1.0f, cell.moisture + v * 0.20f);
        cell.precipitation = std::min(1.0f, cell.precipitation + transpired_precip[i]);

        // 2. Regulación térmica (Termostato Vegetal):
        // - En climas cálidos (temp > 0.55 ~ >15°C): el enfriamiento evaporativo de las hojas
        //   refresca la superficie hasta -3.5°C (-0.064 norm) y amortigua la estacionalidad.
        // - En climas subárticos/boreales (0.16 < temp < 0.34): el dosel oscuro de coníferas
        //   absorbe radiación solar sobre la nieve (efecto albedo), calentando +1.4°C (+0.025 norm).
        if (cell.temperature > 0.55f) {
            float heat_excess = cell.temperature - 0.55f;
            float cooling = v * heat_excess * 0.22f;
            cell.temperature = std::max(0.0f, cell.temperature - cooling);
        } else if (cell.temperature > 0.16f && cell.temperature < 0.34f) {
            float boreal_warming = v * 0.025f;
            cell.temperature = std::min(1.0f, cell.temperature + boreal_warming);
        }

        // El dosel forestal amortigua los extremos térmicos entre verano e invierno
        cell.temp_seasonality *= (1.0f - 0.32f * v);

        // 3. Génesis de suelo orgánico (Humus):
        // Donde se establece vegetación densa (> 0.50), la materia orgánica transforma arenas
        // o suelos jóvenes en suelo franco fértil (LOAM), salvo en lateritas tropicales extremas.
        if (v > 0.50f && cell.soil == SoilType::SAND) {
            cell.soil = SoilType::LOAM;
        }
    }
}

} // namespace Ravis