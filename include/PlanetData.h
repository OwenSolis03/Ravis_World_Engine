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

