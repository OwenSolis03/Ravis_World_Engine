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
    RAINFOREST,
    DESERT,
    SAVANNA
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
    float height = 0.0f; // Normalized height (0.0 to 1.0)
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

    // Topology
    std::vector<size_t> neighbors; // IDs of neighboring cells
};

// Helper function to interpolate normalized height to real-world meters
// e.g., 0.0 -> -50,000m (deepest trench), 1.0 -> 30,000m (highest peak)
inline float lerpHeight(float normalized_height, float min_height = -50000.0f, float max_height = 30000.0f) {
    return min_height + normalized_height * (max_height - min_height);
}

} // namespace Ravis
