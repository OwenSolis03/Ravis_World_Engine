#include "../include/MapExporter.h"
#include "../include/MathUtils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

namespace Ravis {

struct Color { unsigned char r, g, b; };

static Color getCellBiomeColor(const Cell& cell, float sea_level) {
    if (cell.elevation <= sea_level) {
        float depth = 1.0f - std::max(0.0f, std::min(1.0f, (cell.elevation + 10000.0f) / (sea_level + 10000.0f)));
        if (cell.temperature < 0.2f) return { 220, 240, 255 }; // Hielo marino
        return { static_cast<unsigned char>(10 + 30 * (1.0f - depth)),
                 static_cast<unsigned char>(20 + 40 * (1.0f - depth)),
                 static_cast<unsigned char>(100 + 100 * (1.0f - depth)) };
    }

    float r, g, b;
    biomeRGB(cell.biome, r, g, b);
    return { static_cast<unsigned char>(r * 255.0f),
             static_cast<unsigned char>(g * 255.0f),
             static_cast<unsigned char>(b * 255.0f) };
}

// Nearest cell to p, walking the neighbor graph uphill in dot(cell, p) from
// `start` until no neighbor is closer.
static size_t walkToNearestCell(const std::vector<Cell>& cells, size_t start, const Vector3& p) {
    size_t current = start;
    double best = Math::dotProduct(cells[current].position, p);
    bool moved = true;
    while (moved) {
        moved = false;
        for (size_t nid : cells[current].neighbors) {
            double d = Math::dotProduct(cells[nid].position, p);
            if (d > best) {
                best = d;
                current = nid;
                moved = true;
            }
        }
    }
    return current;
}

// Pixel (x, y) samples lat = pi/2 - y/(h-1)*pi, lon = x/(w-1)*2pi - pi.
// The first column is walked top to bottom to seed every row; rows are then
// independent and run in parallel, each pixel starting from its left neighbor.
PixelCellMap MapExporter::buildPixelToCellMap(const GoldbergPolyhedron& planet, int width, int height) {
    const auto& cells = planet.getCells();
    PixelCellMap map(static_cast<size_t>(width) * height);
    if (cells.empty()) return map;

    auto pixelPoint = [&](int x, int y) {
        double lat = Math::PI * 0.5 - (static_cast<double>(y) / (height - 1)) * Math::PI;
        double lon = (static_cast<double>(x) / (width - 1)) * 2.0 * Math::PI - Math::PI;
        return Math::sphericalToCartesian(lat, lon);
    };

    std::vector<size_t> row_seed(height);
    size_t seed = 0;
    for (int y = 0; y < height; ++y) {
        seed = walkToNearestCell(cells, seed, pixelPoint(0, y));
        row_seed[y] = seed;
    }

#pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        size_t current = row_seed[y];
        uint32_t* row = map.data() + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            current = walkToNearestCell(cells, current, pixelPoint(x, y));
            row[x] = static_cast<uint32_t>(current);
        }
    }
    return map;
}

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
        pixels_height[p] = hVal; pixels_height[p + 1] = hVal; pixels_height[p + 2] = hVal;

        unsigned char tVal = static_cast<unsigned char>(cell.temperature * 255.0f);
        pixels_temp[p] = tVal; pixels_temp[p + 1] = 0; pixels_temp[p + 2] = 255 - tVal;

        unsigned char mVal = static_cast<unsigned char>(cell.precipitation * 255.0f);
        pixels_moist[p] = 0; pixels_moist[p + 1] = 0; pixels_moist[p + 2] = mVal;

        Color bCol = getCellBiomeColor(cell, sea_level);
        pixels_biome[p] = bCol.r; pixels_biome[p + 1] = bCol.g; pixels_biome[p + 2] = bCol.b;
    }

    stbi_write_png("height.png", width, height, 3, pixels_height.data(), width * 3);
    stbi_write_png("temperature.png", width, height, 3, pixels_temp.data(), width * 3);
    stbi_write_png("moisture.png", width, height, 3, pixels_moist.data(), width * 3);
    stbi_write_png("biome.png", width, height, 3, pixels_biome.data(), width * 3);
}

std::vector<unsigned char> MapExporter::getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();

    std::vector<float> flows;
    for (const auto& cell : cells) {
        if (cell.elevation > sea_level && cell.river_flow > 0.0f && !cell.is_lake)
            flows.push_back(cell.river_flow);
    }
    float riverThreshold = 0.0f;
    if (!flows.empty()) {
        // Only the percentile is needed, not a full sort.
        auto k = flows.begin() + static_cast<size_t>(flows.size() * 0.975);
        std::nth_element(flows.begin(), k, flows.end());
        riverThreshold = *k;
    }

    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        Color bCol = getCellBiomeColor(cell, sea_level);

        if (cell.is_lake) {
            bCol = {100, 149, 237};
        } else if (cell.river_flow >= riverThreshold && riverThreshold > 0.0f && cell.elevation > sea_level) {
            bCol = {30, 80, 200};
        }

        int p = idx * 3;
        pixels[p] = bCol.r; pixels[p + 1] = bCol.g; pixels[p + 2] = bCol.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();
    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0, 0, 0};
        if (cell.elevation <= sea_level) {
            c = {20, 20, 80};
        } else {
            switch (cell.bedrock) {
                case RockType::BASALT:          c = {30, 30, 30}; break;
                case RockType::GRANITE:         c = {120, 120, 120}; break;
                case RockType::SANDSTONE:       c = {210, 180, 140}; break;
                case RockType::SHALE_LIMESTONE: c = {160, 160, 150}; break;
                case RockType::METAMORPHIC:     c = {200, 200, 255}; break;
            }
        }
        pixels[p] = c.r; pixels[p + 1] = c.g; pixels[p + 2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();
    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0, 0, 0};
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
        pixels[p] = c.r; pixels[p + 1] = c.g; pixels[p + 2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();

    // Calcular umbral de ríos (top 2.5% de caudal) para superponerlos sobre el mapa de alturas
    std::vector<float> flows;
    flows.reserve(cells.size() / 4);
    for (const auto& cell : cells) {
        if (cell.elevation > sea_level && cell.river_flow > 0.0f && !cell.is_lake) {
            flows.push_back(cell.river_flow);
        }
    }
    float riverThreshold = 0.0f;
    if (!flows.empty()) {
        size_t kth = static_cast<size_t>(flows.size() * 0.975);
        std::nth_element(flows.begin(), flows.begin() + kth, flows.end());
        riverThreshold = flows[kth];
    }

    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        Color c = {0, 0, 0};
        float e = cell.elevation;
        if (e <= sea_level) {
            float depth = std::max(0.0f, std::min(1.0f, (e + 11000.0f) / (sea_level + 11000.0f)));
            c.r = 0;
            c.g = static_cast<unsigned char>(100 * depth);
            c.b = static_cast<unsigned char>(50 + 150 * depth);
            if (cell.temperature < 0.2f) {
                c.r = 220; c.g = 240; c.b = 255; // Hielo marino
            }
        } else {
            float land = (e - sea_level);
            if (land < 1000.0f) {
                float t = land / 1000.0f;
                c.r = static_cast<unsigned char>(34 * (1 - t) + 173 * t);
                c.g = static_cast<unsigned char>(139 * (1 - t) + 255 * t);
                c.b = static_cast<unsigned char>(34 * (1 - t) + 47 * t);
            } else if (land < 3000.0f) {
                float t = (land - 1000.0f) / 2000.0f;
                c.r = static_cast<unsigned char>(173 * (1 - t) + 139 * t);
                c.g = static_cast<unsigned char>(255 * (1 - t) + 69 * t);
                c.b = static_cast<unsigned char>(47 * (1 - t) + 19 * t);
            } else {
                float t = std::min(1.0f, (land - 3000.0f) / 5000.0f);
                c.r = static_cast<unsigned char>(139 * (1 - t) + 60 * t);
                c.g = static_cast<unsigned char>(69 * (1 - t) + 30 * t);
                c.b = static_cast<unsigned char>(19 * (1 - t) + 10 * t);
            }

            if (cell.temperature < 0.2f) {
                c.r = 255; c.g = 255; c.b = 255; // Glaciares / Nieve
            }

            // Superponer lagos y ríos por encima de la altura y la nieve para debugging
            if (cell.is_lake) {
                c = {0, 190, 240}; // Cian brillante para lagos
            } else if (riverThreshold > 0.0f && cell.river_flow >= riverThreshold) {
                c = {20, 70, 235}; // Azul eléctrico para ríos
            }
        }

        pixels[p] = c.r; pixels[p + 1] = c.g; pixels[p + 2] = c.b;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();
    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        unsigned char tVal = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, cell.temperature)) * 255.0f);
        pixels[p] = tVal;
        pixels[p + 1] = 0;
        pixels[p + 2] = 255 - tVal;
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();
    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;
        unsigned char mVal = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, cell.precipitation)) * 255.0f);
        pixels[p] = 0;
        pixels[p + 1] = static_cast<unsigned char>(mVal * 0.35f);
        pixels[p + 2] = mVal;
    }
    return pixels;
}

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

std::vector<unsigned char> MapExporter::getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap) {
    const auto& cells = planet.getCells();

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
        // Only the percentile is needed, not a full sort.
        auto k = flows.begin() + static_cast<size_t>(flows.size() * 0.90);
        std::nth_element(flows.begin(), k, flows.end());
        riverThreshold = *k;
    }

    const int n = width * height;
    std::vector<unsigned char> pixels(static_cast<size_t>(n) * 3);
#pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        const Cell& cell = cells[cellMap[idx]];
        int p = idx * 3;

        if (cell.elevation <= sea_level) {
            pixels[p] = 15; pixels[p + 1] = 15; pixels[p + 2] = 60;
        } else if (cell.is_lake) {
            pixels[p] = 0; pixels[p + 1] = 200; pixels[p + 2] = 220;
        } else if (cell.river_flow >= riverThreshold && riverThreshold > 0.0f) {
            float intensity = std::min(1.0f, cell.river_flow / maxFlow);
            unsigned char b = static_cast<unsigned char>(100 + 155 * intensity);
            pixels[p] = 20; pixels[p + 1] = 50; pixels[p + 2] = b;
        } else {
            pixels[p] = 51; pixels[p + 1] = 51; pixels[p + 2] = 51;
        }
    }
    return pixels;
}

} // namespace Ravis