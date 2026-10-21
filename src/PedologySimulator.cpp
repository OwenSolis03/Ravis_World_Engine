#include "../include/PedologySimulator.h"
#include <cmath>

namespace Ravis {

PedologySimulator::PedologySimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void PedologySimulator::classifyBedrock(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();

    for (auto& cell : cells) {
        // Celdas sumergidas bajo el nivel del mar efectivo
        if (cell.elevation <= sea) {
            if (cell.is_oceanic) {
                cell.bedrock = (cell.elevation > sea - 1400.0f)
                                   ? RockType::SHALE_LIMESTONE
                                   : RockType::BASALT;
            } else {
                // Plataforma continental sumergida o mar epicontinental
                cell.bedrock = RockType::SHALE_LIMESTONE;
            }
            continue;
        }

        const float elevAbove = cell.elevation - sea;

        // Islas volcánicas oceánicas o terrenos emergidos sobre placas oceánicas
        if (cell.is_oceanic) {
            if (elevAbove > 2200.0f && cell.crustal_thickness > 35.0f) {
                cell.bedrock = RockType::METAMORPHIC;
            } else if (elevAbove < 180.0f && (cell.is_lake || cell.precipitation > 0.35f)) {
                cell.bedrock = RockType::SHALE_LIMESTONE; // Arrecifes / carbonatos costeros
            } else {
                cell.bedrock = RockType::BASALT; // Isla volcánica basáltica (ej. Hawái / Islandia)
            }
            continue;
        }

        // Cinturones orogénicos continentales y escudos engrosados
        if (elevAbove > 1600.0f || cell.crustal_thickness > 46.0f) {
            cell.bedrock = RockType::METAMORPHIC;
            continue;
        }

        // Cuencas sedimentarias bajas
        if (elevAbove < 550.0f) {
            if (cell.precipitation < 0.18f) {
                cell.bedrock = RockType::SANDSTONE; // Cuenca clástica árida
            } else if (cell.is_lake || cell.river_flow > 0.05f || cell.temperature > 0.52f) {
                cell.bedrock = RockType::SHALE_LIMESTONE; // Llanura aluvial / lutitas y calizas
            } else {
                cell.bedrock = RockType::SANDSTONE;
            }
            continue;
        }

        // Mesetas de altitud media con meteorización húmeda
        if (cell.precipitation > 0.25f) {
            cell.bedrock = RockType::SANDSTONE;
            continue;
        }

        // Basamento cratónico expuesto (escudos graníticos)
        cell.bedrock = RockType::GRANITE;
    }
}

void PedologySimulator::generateSoils(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();

    for (auto& cell : cells) {
        if (cell.elevation <= sea) {
            cell.soil = SoilType::NONE;
            continue;
        }

        const float elevAbove = cell.elevation - sea;

        // Casquetes de hielo y cumbres nivales de roca desnuda
        if (cell.biome == BiomeType::ICE || cell.temperature < 0.14f || elevAbove > 4200.0f) {
            cell.soil = SoilType::NONE;
            continue;
        }

        if (cell.biome == BiomeType::TUNDRA) {
            cell.soil = SoilType::LOAM;
            continue;
        }

        bool forest = (cell.biome == BiomeType::RAINFOREST ||
                       cell.biome == BiomeType::TEMPERATE_RAINFOREST ||
                       cell.biome == BiomeType::TEMPERATE_FOREST ||
                       cell.biome == BiomeType::BOREAL_FOREST ||
                       cell.biome == BiomeType::TROPICAL_DRY_FOREST);

        if (elevAbove < 95.0f && !forest && !cell.is_lake) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // Desiertos y matorrales hiperáridos -> Aridisoles arenosos
        if (cell.biome == BiomeType::DESERT || cell.precipitation < 0.10f) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // Climas tropicales cálidos y húmedos -> Oxisoles/Lateritas ricas en arcilla
        if (cell.temperature > 0.52f && cell.precipitation > 0.42f) {
            cell.soil = SoilType::CLAY;
            continue;
        }

        if (cell.bedrock == RockType::SHALE_LIMESTONE ||
            (cell.precipitation > 0.38f &&
             (cell.bedrock == RockType::BASALT || cell.bedrock == RockType::GRANITE))) {
            cell.soil = SoilType::CLAY;
            continue;
        }

        cell.soil = SoilType::LOAM;
    }
}

} // namespace Ravis