#include "../include/PedologySimulator.h"
#include <cmath>

namespace Ravis {

PedologySimulator::PedologySimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void PedologySimulator::generateSoils(const SimulationParameters& params) {
    auto& cells = planet.getCells();

    for (auto& cell : cells) {
        // No soil underwater
        if (cell.height <= params.sea_level) {
            cell.soil = SoilType::NONE;
            continue;
        }

        // Base rules for soil generation

        // 1. Coastlines are mostly sand
        if (cell.height <= params.sea_level + 0.02f) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // 2. High organic material biomes generate Loam
        if (cell.biome == BiomeType::TEMPERATE_FOREST || 
            cell.biome == BiomeType::RAINFOREST || 
            cell.biome == BiomeType::GRASSLAND ||
            cell.biome == BiomeType::SAVANNA) {
            cell.soil = SoilType::LOAM;
            continue;
        }

        // 3. Arid regions over sandstone are sandy
        if (cell.precipitation < 0.2f && cell.bedrock == RockType::SANDSTONE) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // 4. Heavily weathered igneous rock or shale turns to clay
        if ((cell.precipitation > 0.5f && (cell.bedrock == RockType::BASALT || cell.bedrock == RockType::GRANITE)) ||
            cell.bedrock == RockType::SHALE_LIMESTONE) {
            cell.soil = SoilType::CLAY;
            continue;
        }

        // Default fallback based on biome
        if (cell.biome == BiomeType::DESERT) {
            cell.soil = SoilType::SAND;
        } else if (cell.biome == BiomeType::ICE || cell.biome == BiomeType::TUNDRA) {
            cell.soil = SoilType::NONE; // Permafrost/bare rock
        } else {
            cell.soil = SoilType::NONE;
        }
    }
}

} // namespace Ravis
