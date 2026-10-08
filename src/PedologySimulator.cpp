#include "../include/PedologySimulator.h"
#include <cmath>

namespace Ravis {

PedologySimulator::PedologySimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void PedologySimulator::classifyBedrock(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();

    for (auto& cell : cells) {
        if (cell.is_oceanic) {
            // Deep sea floor is fresh basalt; the shallow shelf accumulates a
            // marine sediment (carbonate / clastic) cover.
            cell.bedrock = (cell.elevation > sea - 1500.0f)
                               ? RockType::SHALE_LIMESTONE
                               : RockType::BASALT;
            continue;
        }

        // Submerged continental crust: shelf / inland sea -> marine sediment.
        if (cell.elevation <= sea) {
            cell.bedrock = RockType::SHALE_LIMESTONE;
            continue;
        }

        const float elevAbove = cell.elevation - sea;

        // Orogenic belts / uplifted shield: high ground or thick crustal roots
        // expose metamorphic and high-grade rock.
        if (elevAbove > 1600.0f || cell.crustal_thickness > 46.0f) {
            cell.bedrock = RockType::METAMORPHIC;
            continue;
        }

        // Low continental basins.
        if (elevAbove < 600.0f) {
            if (cell.precipitation < 0.18f) {
                cell.bedrock = RockType::SANDSTONE;       // arid clastic basin / dunes
            } else if (cell.is_lake || cell.river_flow > 0.05f || cell.temperature > 0.50f) {
                cell.bedrock = RockType::SHALE_LIMESTONE; // floodplain mud / warm carbonate
            } else {
                cell.bedrock = RockType::SANDSTONE;
            }
            continue;
        }

        // Mid-elevation uplands with any real rainfall: sedimentary veneer.
        if (cell.precipitation > 0.22f) {
            cell.bedrock = RockType::SANDSTONE;
            continue;
        }

        cell.bedrock = RockType::GRANITE; // exposed continental basement
    }
}

void PedologySimulator::generateSoils(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();

    for (auto& cell : cells) {
        // No soil underwater.
        if (cell.elevation <= sea) {
            cell.soil = SoilType::NONE;
            continue;
        }

        const float elevAbove = cell.elevation - sea;

        // Ice sheets and the coldest bare rock have no real soil profile.
        if (cell.biome == BiomeType::ICE || cell.temperature < 0.15f) {
            cell.soil = SoilType::NONE;
            continue;
        }

        // Tundra keeps a thin peaty / cryic profile.
        if (cell.biome == BiomeType::TUNDRA) {
            cell.soil = SoilType::LOAM;
            continue;
        }

        // Narrow coastal fringe (non-forest) is beach / dune sand.
        bool forest = (cell.biome == BiomeType::RAINFOREST ||
                       cell.biome == BiomeType::TEMPERATE_RAINFOREST ||
                       cell.biome == BiomeType::TEMPERATE_FOREST ||
                       cell.biome == BiomeType::BOREAL_FOREST ||
                       cell.biome == BiomeType::TROPICAL_DRY_FOREST);
        if (elevAbove < 120.0f && !forest) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // Only truly arid land -> sand (aridisol / erg). Semi-arid grassland
        // keeps a loam profile.
        if (cell.precipitation < 0.11f || cell.biome == BiomeType::DESERT) {
            cell.soil = SoilType::SAND;
            continue;
        }

        // Hot & very wet -> deeply leached clay (oxisol / laterite).
        if (cell.temperature > 0.5f && cell.precipitation > 0.45f) {
            cell.soil = SoilType::CLAY;
            continue;
        }

        // Clay also over marine sediment, and over igneous rock in wet climates.
        if (cell.bedrock == RockType::SHALE_LIMESTONE ||
            (cell.precipitation > 0.4f &&
             (cell.bedrock == RockType::BASALT || cell.bedrock == RockType::GRANITE))) {
            cell.soil = SoilType::CLAY;
            continue;
        }

        // Everything else vegetated -> loam (grassland, steppe, savanna,
        // mediterranean, forests, thorn scrub).
        cell.soil = SoilType::LOAM;
    }
}

} // namespace Ravis
