#include "../include/MapExporter.h"
#include "../include/MathUtils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

namespace Ravis {

struct Color { unsigned char r, g, b; };

struct BiomePoint {
    float temp;
    float precip;
    Color color;
};

const BiomePoint biomes[] = {
    // Cold
    { 0.2f, 0.1f, {160, 160, 120} }, // Cold Desert
    { 0.2f, 0.4f, {140, 150, 100} }, // Steppe / Tundra
    { 0.3f, 0.7f, {90, 120, 90} },   // Boreal Forest
    { 0.3f, 0.9f, {50, 90, 60} },    // Temperate Rainforest (Cold)

    // Temperate
    { 0.5f, 0.1f, {210, 180, 140} }, // Desert
    { 0.6f, 0.3f, {180, 180, 90} },  // Mediterranean / Shrubland
    { 0.5f, 0.6f, {120, 180, 90} },  // Temperate Forest
    { 0.6f, 0.9f, {34, 139, 34} },   // Temperate Rainforest

    // Tropical
    { 0.9f, 0.1f, {237, 201, 175} }, // Hot Desert
    { 0.8f, 0.3f, {200, 180, 100} }, // Thorn Scrub
    { 0.9f, 0.5f, {154, 205, 50} },  // Savanna
    { 0.8f, 0.7f, {100, 160, 40} },  // Tropical Dry Forest
    { 0.9f, 0.9f, {0, 100, 0} }      // Tropical Rainforest
};

Color getBiomeColor(float temp, float precip, float elevation, float sea_level) {
    if (elevation <= sea_level) {
        float depth = 1.0f - std::max(0.0f, std::min(1.0f, (elevation + 10000.0f) / (sea_level + 10000.0f)));
        // If it's very cold, ocean freezes over
        if (temp < 0.2f) return { 220, 240, 255 }; // Sea ice
        return { static_cast<unsigned char>(10 + 30 * (1.0f - depth)),
                 static_cast<unsigned char>(20 + 40 * (1.0f - depth)),
                 static_cast<unsigned char>(100 + 100 * (1.0f - depth)) };
    }
    
    // Explicit ice overlay for land
    if (temp < 0.2f) return { 240, 240, 240 }; // Ice cap / Glacier
    
    // Euclidean Whittaker diagram search
    float min_dist = 1e10f;
    Color best_color = {0, 0, 0};
    
    for (const auto& b : biomes) {
        float dt = temp - b.temp;
        float dp = precip - b.precip;
        float dist = dt*dt + dp*dp;
        if (dist < min_dist) {
            min_dist = dist;
            best_color = b.color;
        }
    }
    return best_color;
}

// External CUDA function declaration
extern void buildPixelToCellMapCUDA(int width, int height, int num_cells, const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z, std::vector<size_t>& map);

// ============================================================================
// Pixel → Cell Lookup Table (O(pixels * cells) — accelerated with CUDA)
// ============================================================================
std::vector<size_t> MapExporter::buildPixelToCellMap(const GoldbergPolyhedron& planet, int width, int height) {
    const auto& cells = planet.getCells();
    std::vector<size_t> map(width * height);
    
    int num_cells = cells.size();
    std::vector<float> pos_x(num_cells);
    std::vector<float> pos_y(num_cells);
    std::vector<float> pos_z(num_cells);
    
    for (int i = 0; i < num_cells; ++i) {
        pos_x[i] = cells[i].position.x;
        pos_y[i] = cells[i].position.y;
        pos_z[i] = cells[i].position.z;
    }
    
    buildPixelToCellMapCUDA(width, height, num_cells, pos_x, pos_y, pos_z, map);

    return map;
}

// ============================================================================
// Fast versions (take precomputed cellMap) — O(pixels) each
// ============================================================================

void MapExporter::exportAllMaps(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    auto cellMap = buildPixelToCellMap(planet, width, height);
    const auto& cells = planet.getCells();

    std::vector<unsigned char> pixels_height(width * height * 3);
    std::vector<unsigned char> pixels_temp(width * height * 3);
    std::vector<unsigned char> pixels_moist(width * height * 3);
    std::vector<unsigned char> pixels_biome(width * height * 3);

    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;

        float normalized = (cell.elevation + 10000.0f) / 20000.0f;
        unsigned char hVal = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, normalized * 255.0f)));
        pixels_height[p] = hVal; pixels_height[p+1] = hVal; pixels_height[p+2] = hVal;

        unsigned char tVal = static_cast<unsigned char>(cell.temperature * 255.0f);
        pixels_temp[p] = tVal; pixels_temp[p+1] = 0; pixels_temp[p+2] = 255 - tVal;

        unsigned char mVal = static_cast<unsigned char>(cell.precipitation * 255.0f);
        pixels_moist[p] = 0; pixels_moist[p+1] = 0; pixels_moist[p+2] = mVal;

        Color bCol = getBiomeColor(cell.temperature, cell.precipitation, cell.elevation, sea_level);
        pixels_biome[p] = bCol.r; pixels_biome[p+1] = bCol.g; pixels_biome[p+2] = bCol.b;
    }

    stbi_write_png("height.png", width, height, 3, pixels_height.data(), width * 3);
    stbi_write_png("temperature.png", width, height, 3, pixels_temp.data(), width * 3);
    stbi_write_png("moisture.png", width, height, 3, pixels_moist.data(), width * 3);
    stbi_write_png("biome.png", width, height, 3, pixels_biome.data(), width * 3);
}

std::vector<unsigned char> MapExporter::getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    
    // Compute river threshold (top 5% of flow)
    std::vector<float> flows;
    for (const auto& cell : cells) {
        if (cell.elevation > sea_level && cell.river_flow > 0.0f && !cell.is_lake)
            flows.push_back(cell.river_flow);
    }
    float riverThreshold = 0.0f;
    if (!flows.empty()) {
        std::sort(flows.begin(), flows.end());
        riverThreshold = flows[static_cast<size_t>(flows.size() * 0.95)];
    }
    
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        Color bCol = getBiomeColor(cell.temperature, cell.precipitation, cell.elevation, sea_level);
        
        // Overlay lakes
        if (cell.is_lake) {
            bCol = {100, 149, 237}; // Cornflower blue
        }
        // Overlay rivers
        else if (cell.river_flow >= riverThreshold && riverThreshold > 0.0f && cell.elevation > sea_level) {
            bCol = {30, 80, 200}; // River blue
        }
        
        int p = idx * 3;
        pixels[p] = bCol.r; pixels[p+1] = bCol.g; pixels[p+2] = bCol.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0,0,0};
        if (cell.elevation <= sea_level) {
            c = {20, 20, 80};
        } else {
            switch (cell.bedrock) {
                case RockType::BASALT: c = {30, 30, 30}; break;
                case RockType::GRANITE: c = {120, 120, 120}; break;
                case RockType::SANDSTONE: c = {210, 180, 140}; break;
                case RockType::SHALE_LIMESTONE: c = {160, 160, 150}; break;
                case RockType::METAMORPHIC: c = {200, 200, 255}; break;
            }
        }
        pixels[p] = c.r; pixels[p+1] = c.g; pixels[p+2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0,0,0};
        if (cell.elevation <= sea_level) {
            c = {20, 20, 80};
        } else {
            switch (cell.soil) {
                case SoilType::NONE: c = {100, 100, 100}; break;
                case SoilType::SAND: c = {240, 230, 140}; break;
                case SoilType::CLAY: c = {180, 100, 50}; break;
                case SoilType::LOAM: c = {80, 40, 20}; break;
            }
        }
        pixels[p] = c.r; pixels[p+1] = c.g; pixels[p+2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0,0,0};
        float e = cell.elevation;
        if (e <= sea_level) {
            float depth = std::max(0.0f, std::min(1.0f, (e + 11000.0f) / (sea_level + 11000.0f)));
            c.r = static_cast<unsigned char>(0);
            c.g = static_cast<unsigned char>(100 * depth);
            c.b = static_cast<unsigned char>(50 + 150 * depth);
        } else {
            // Above sea level: interpolate from green -> yellow-green -> brown -> dark brown
            float land = (e - sea_level);
            if (land < 1000.0f) {
                float t = land / 1000.0f;
                c.r = static_cast<unsigned char>(34 * (1-t) + 173 * t);
                c.g = static_cast<unsigned char>(139 * (1-t) + 255 * t);
                c.b = static_cast<unsigned char>(34 * (1-t) + 47 * t);
            } else if (land < 3000.0f) {
                float t = (land - 1000.0f) / 2000.0f;
                c.r = static_cast<unsigned char>(173 * (1-t) + 139 * t);
                c.g = static_cast<unsigned char>(255 * (1-t) + 69 * t);
                c.b = static_cast<unsigned char>(47 * (1-t) + 19 * t);
            } else {
                float t = std::min(1.0f, (land - 3000.0f) / 5000.0f);
                c.r = static_cast<unsigned char>(139 * (1-t) + 60 * t);
                c.g = static_cast<unsigned char>(69 * (1-t) + 30 * t);
                c.b = static_cast<unsigned char>(19 * (1-t) + 10 * t);
            }
        }
        
        if (cell.temperature < 0.2f) {
            c.r = 255;
            c.g = 255;
            c.b = 255;
        }
        
        pixels[p] = c.r; pixels[p+1] = c.g; pixels[p+2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        unsigned char tVal = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, cell.temperature)) * 255.0f);
        pixels[p] = tVal; 
        pixels[p+1] = 0; 
        pixels[p+2] = 255 - tVal;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        unsigned char mVal = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, cell.precipitation)) * 255.0f);
        pixels[p] = 0; 
        pixels[p+1] = 0; 
        pixels[p+2] = mVal;
    }
    return pixels;
}


// ============================================================================
// Convenience versions (build cellMap internally — for single-use calls)
// ============================================================================

std::vector<unsigned char> MapExporter::getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getBiomePixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getLithologyPixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getPedologyPixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getHeightPixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getHydrologyPixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getTemperaturePixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}
std::vector<unsigned char> MapExporter::getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    return getMoisturePixels(planet, width, height, sea_level, buildPixelToCellMap(planet, width, height));
}

// ============================================================================
// Hydrology Map — rivers, lakes, and drainage
// ============================================================================
std::vector<unsigned char> MapExporter::getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap) {
    const auto& cells = planet.getCells();
    
    // Compute max flow for normalization and river threshold
    float maxFlow = 0.01f;
    std::vector<float> flows;
    for (const auto& cell : cells) {
        if (cell.elevation > sea_level && !cell.is_lake) {
            if (cell.river_flow > maxFlow) maxFlow = cell.river_flow;
            if (cell.river_flow > 0.0f) flows.push_back(cell.river_flow);
        }
    }
    float riverThreshold = 0.0f;
    if (!flows.empty()) {
        std::sort(flows.begin(), flows.end());
        riverThreshold = flows[static_cast<size_t>(flows.size() * 0.90)]; // Top 10% for hydro map
    }
    
    std::vector<unsigned char> pixels(width * height * 3);
    for (int idx = 0; idx < width * height; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        
        if (cell.elevation <= sea_level) {
            // Ocean: dark blue
            pixels[p] = 15; pixels[p+1] = 15; pixels[p+2] = 60;
        } else if (cell.is_lake) {
            // Lake: cyan
            pixels[p] = 0; pixels[p+1] = 200; pixels[p+2] = 220;
        } else if (cell.river_flow >= riverThreshold && riverThreshold > 0.0f) {
            // River: blue intensity proportional to flow
            float intensity = std::min(1.0f, cell.river_flow / maxFlow);
            unsigned char b = static_cast<unsigned char>(100 + 155 * intensity);
            pixels[p] = 20; pixels[p+1] = 50; pixels[p+2] = b;
        } else {
            // Land: dark gray
            pixels[p] = 51;
            pixels[p+1] = 51;
            pixels[p+2] = 51;
        }
    }
    return pixels;
}

} // namespace Ravis
