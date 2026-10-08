#pragma once

#include <vector>
#include <cstddef>

namespace Ravis {

// A simple 3D vector for positions and velocities
struct Vector3 {
    double x, y, z;

    Vector3() : x(0), y(0), z(0) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
};

enum class BiomeType {
    OCEAN,
    ICE,
    TUNDRA,
    BOREAL_FOREST,
    TEMPERATE_FOREST,
    GRASSLAND,
    STEPPE,
    RAINFOREST,
    TEMPERATE_RAINFOREST,
    TROPICAL_DRY_FOREST,
    MEDITERRANEAN,
    DESERT,
    SAVANNA,
    THORN_SCRUB
};

enum class RockType {
    BASALT,
    GRANITE,
    SANDSTONE,
    SHALE_LIMESTONE,
    METAMORPHIC
};

enum class SoilType {
    NONE,
    SAND,
    CLAY,
    LOAM
};

// ---------------------------------------------------------------------------
// Biome classification — single source of truth
// ---------------------------------------------------------------------------
// A Whittaker-style diagram keyed on mean temperature (deg C) and annual
// precipitation. `precip01` is the engine's normalized precipitation (0..1);
// it is scaled to ~cm/yr internally. Used by AtmosphereSimulator (writes
// Cell::biome), the 2D MapExporter and the 3D globe render so all three agree.
inline BiomeType classifyBiome(float tempC, float precip01) {
    const float p = precip01 * 400.0f; // ~cm/yr (0.25 normalized -> 100 cm/yr)

    // Very cold: precipitation is nearly irrelevant.
    if (tempC < -8.0f) return BiomeType::ICE;
    if (tempC < -2.0f) return BiomeType::TUNDRA;

    if (tempC < 5.0f) { // boreal / subarctic
        if (p < 25.0f) return BiomeType::TUNDRA;
        return BiomeType::BOREAL_FOREST;
    }

    if (tempC < 12.0f) { // cool temperate
        if (p < 30.0f)  return BiomeType::STEPPE;
        if (p < 100.0f) return BiomeType::TEMPERATE_FOREST;
        return BiomeType::TEMPERATE_RAINFOREST;
    }

    if (tempC < 20.0f) { // warm temperate
        if (p < 25.0f)  return BiomeType::DESERT;
        if (p < 50.0f)  return BiomeType::GRASSLAND;
        if (p < 90.0f)  return BiomeType::MEDITERRANEAN;
        if (p < 200.0f) return BiomeType::TEMPERATE_FOREST;
        return BiomeType::TEMPERATE_RAINFOREST;
    }

    // Tropical (tempC >= 20)
    if (p < 25.0f)  return BiomeType::DESERT;
    if (p < 60.0f)  return BiomeType::THORN_SCRUB;
    if (p < 120.0f) return BiomeType::SAVANNA;
    if (p < 220.0f) return BiomeType::TROPICAL_DRY_FOREST;
    return BiomeType::RAINFOREST;
}

// Canonical land-biome colour (linear 0..1 RGB). OCEAN is handled by the
// callers (depth gradient), so it falls through to a neutral grey here.
inline void biomeRGB(BiomeType b, float& r, float& g, float& bl) {
    switch (b) {
    case BiomeType::ICE:                  r = 0.96f; g = 0.96f; bl = 0.98f; break;
    case BiomeType::TUNDRA:               r = 0.59f; g = 0.63f; bl = 0.55f; break;
    case BiomeType::BOREAL_FOREST:        r = 0.27f; g = 0.39f; bl = 0.29f; break;
    case BiomeType::TEMPERATE_FOREST:     r = 0.27f; g = 0.55f; bl = 0.24f; break;
    case BiomeType::GRASSLAND:            r = 0.67f; g = 0.78f; bl = 0.39f; break;
    case BiomeType::STEPPE:               r = 0.78f; g = 0.76f; bl = 0.47f; break;
    case BiomeType::RAINFOREST:           r = 0.08f; g = 0.43f; bl = 0.12f; break;
    case BiomeType::TEMPERATE_RAINFOREST: r = 0.16f; g = 0.43f; bl = 0.27f; break;
    case BiomeType::TROPICAL_DRY_FOREST:  r = 0.43f; g = 0.63f; bl = 0.24f; break;
    case BiomeType::MEDITERRANEAN:        r = 0.71f; g = 0.69f; bl = 0.37f; break;
    case BiomeType::DESERT:               r = 0.88f; g = 0.78f; bl = 0.59f; break;
    case BiomeType::SAVANNA:              r = 0.75f; g = 0.71f; bl = 0.35f; break;
    case BiomeType::THORN_SCRUB:          r = 0.78f; g = 0.67f; bl = 0.43f; break;
    case BiomeType::OCEAN:
    default:                              r = 0.40f; g = 0.40f; bl = 0.40f; break;
    }
}

// Represents a single cell (hexagon or pentagon) on the planetary surface
struct Cell {
    size_t id;
    double latitude;
    double longitude;
    Vector3 position; // 3D cartesian coordinates for easy distance math

    // Attributes
    float elevation = 0.0f; // Elevation in meters. Negative = below sea level.
    bool is_oceanic = false; // Crust type: true for oceanic, false for continental
    int plate_id = -1;
    Vector3 plate_velocity = Vector3(0,0,0);
    float temperature = 0.0f; // Normalized 0.0 (Cold) to 1.0 (Hot)
    float moisture = 0.0f; // Normalized 0.0 (Dry) to 1.0 (Wet)
    float precipitation = 0.0f; // Amount of rain received
    Vector3 wind_velocity = Vector3(0,0,0); // Direction of prevailing winds
    
    // Geology
    RockType bedrock = RockType::BASALT;
    SoilType soil = SoilType::NONE;
    BiomeType biome = BiomeType::OCEAN;

    // Geophysics
    float crustal_age = 0.0f;        // Ma — older oceanic crust is denser and subducts
    float crustal_thickness = 35.0f; // km — continental ~35km, oceanic ~7km
    float sediment_depth = 0.0f;     // Accumulated sedimentary layer thickness

    // Hydrology
    bool is_lake = false;            // True if this cell is part of a lake
    float river_flow = 0.0f;         // Accumulated water flow from upstream cells
    size_t downstream_id = 0;        // ID of the cell this flows into (steepest descent)

    // Topology
    std::vector<size_t> neighbors; // IDs of neighboring cells
};

} // namespace Ravis

