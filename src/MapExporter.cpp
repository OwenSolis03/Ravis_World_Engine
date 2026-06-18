#include "../include/MapExporter.h"
#include "../include/MathUtils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

#include <vector>
#include <iostream>
#include <limits>

namespace Ravis {

struct Color { unsigned char r, g, b; };

Color getBiomeColor(float temp, float precip, float height, float sea_level) {
    if (height <= sea_level) {
        // Ocean depth coloring
        float depth = height / sea_level;
        return { static_cast<unsigned char>(10 + 30 * depth), static_cast<unsigned char>(20 + 40 * depth), static_cast<unsigned char>(100 + 100 * depth) };
    }
    
    // Land Biomes (Whittaker simplified)
    if (temp < 0.2f) return { 240, 240, 240 }; // Ice/Tundra
    if (temp < 0.4f) {
        if (precip < 0.3f) return { 160, 160, 120 }; // Cold Desert
        if (precip < 0.6f) return { 90, 120, 90 }; // Taiga/Boreal
        return { 50, 100, 50 }; // Temperate Rain Forest
    }
    if (temp < 0.7f) {
        if (precip < 0.2f) return { 210, 180, 140 }; // Temperate Desert
        if (precip < 0.5f) return { 120, 180, 90 }; // Grassland
        return { 34, 139, 34 }; // Temperate Forest
    }
    // Hot
    if (precip < 0.2f) return { 237, 201, 175 }; // Subtropical Desert
    if (precip < 0.6f) return { 154, 205, 50 }; // Savanna
    return { 0, 100, 0 }; // Tropical Rain Forest
}

void MapExporter::exportAllMaps(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    std::vector<unsigned char> pixels_height(width * height * 3);
    std::vector<unsigned char> pixels_temp(width * height * 3);
    std::vector<unsigned char> pixels_moist(width * height * 3);
    std::vector<unsigned char> pixels_biome(width * height * 3);

    const auto& cells = planet.getCells();

    for (int y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / (height - 1);
        double lat = Math::PI / 2.0 - v * Math::PI;

        for (int x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / (width - 1);
            double lon = u * 2.0 * Math::PI - Math::PI;

            Vector3 pixelPos = Math::sphericalToCartesian(lat, lon);

            double minDist = std::numeric_limits<double>::max();
            const Cell* closestCell = nullptr;

            for (const auto& cell : cells) {
                double dist = Math::distance(pixelPos, cell.position);
                if (dist < minDist) {
                    minDist = dist;
                    closestCell = &cell;
                }
            }

            int pixelIdx = (y * width + x) * 3;

            if (closestCell) {
                // Height (Grayscale)
                unsigned char hVal = static_cast<unsigned char>(closestCell->height * 255.0f);
                pixels_height[pixelIdx] = hVal; pixels_height[pixelIdx+1] = hVal; pixels_height[pixelIdx+2] = hVal;

                // Temp (Blue to Red)
                unsigned char tVal = static_cast<unsigned char>(closestCell->temperature * 255.0f);
                pixels_temp[pixelIdx] = tVal; pixels_temp[pixelIdx+1] = 0; pixels_temp[pixelIdx+2] = 255 - tVal;

                // Moisture (Black to Blue)
                unsigned char mVal = static_cast<unsigned char>(closestCell->precipitation * 255.0f);
                pixels_moist[pixelIdx] = 0; pixels_moist[pixelIdx+1] = 0; pixels_moist[pixelIdx+2] = mVal;

                // Biome Map
                Color bCol = getBiomeColor(closestCell->temperature, closestCell->precipitation, closestCell->height, sea_level);
                pixels_biome[pixelIdx] = bCol.r; pixels_biome[pixelIdx+1] = bCol.g; pixels_biome[pixelIdx+2] = bCol.b;
            }
        }
    }

    stbi_write_png("height.png", width, height, 3, pixels_height.data(), width * 3);
    stbi_write_png("temperature.png", width, height, 3, pixels_temp.data(), width * 3);
    stbi_write_png("moisture.png", width, height, 3, pixels_moist.data(), width * 3);
    stbi_write_png("biome.png", width, height, 3, pixels_biome.data(), width * 3);
}

std::vector<unsigned char> MapExporter::getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    std::vector<unsigned char> pixels_biome(width * height * 3);
    const auto& cells = planet.getCells();

    for (int y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / (height - 1);
        double lat = Math::PI / 2.0 - v * Math::PI;

        for (int x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / (width - 1);
            double lon = u * 2.0 * Math::PI - Math::PI;

            Vector3 pixelPos = Math::sphericalToCartesian(lat, lon);

            double minDist = std::numeric_limits<double>::max();
            const Cell* closestCell = nullptr;

            for (const auto& cell : cells) {
                double dist = Math::distance(pixelPos, cell.position);
                if (dist < minDist) {
                    minDist = dist;
                    closestCell = &cell;
                }
            }

            int pixelIdx = (y * width + x) * 3;

            if (closestCell) {
                Color bCol = getBiomeColor(closestCell->temperature, closestCell->precipitation, closestCell->height, sea_level);
                pixels_biome[pixelIdx] = bCol.r; pixels_biome[pixelIdx+1] = bCol.g; pixels_biome[pixelIdx+2] = bCol.b;
            }
        }
    }
    return pixels_biome;
}

std::vector<unsigned char> MapExporter::getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    std::vector<unsigned char> pixels(width * height * 3);
    const auto& cells = planet.getCells();

    for (int y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / (height - 1);
        double lat = Math::PI / 2.0 - v * Math::PI;

        for (int x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / (width - 1);
            double lon = u * 2.0 * Math::PI - Math::PI;

            Vector3 pixelPos = Math::sphericalToCartesian(lat, lon);
            double minDist = std::numeric_limits<double>::max();
            const Cell* closestCell = nullptr;

            for (const auto& cell : cells) {
                double dist = Math::distance(pixelPos, cell.position);
                if (dist < minDist) { minDist = dist; closestCell = &cell; }
            }

            int pixelIdx = (y * width + x) * 3;
            if (closestCell) {
                Color c = {0,0,0};
                if (closestCell->height <= sea_level) {
                    c = {20, 20, 80}; // Ocean
                } else {
                    switch (closestCell->bedrock) {
                        case RockType::BASALT: c = {30, 30, 30}; break; // Black/Dark Grey
                        case RockType::GRANITE: c = {120, 120, 120}; break; // Light Grey
                        case RockType::SANDSTONE: c = {210, 180, 140}; break; // Tan
                        case RockType::SHALE_LIMESTONE: c = {160, 160, 150}; break; // Pale Grey/Green
                        case RockType::METAMORPHIC: c = {200, 200, 255}; break; // Shiny White/Blue
                    }
                }
                pixels[pixelIdx] = c.r; pixels[pixelIdx+1] = c.g; pixels[pixelIdx+2] = c.b;
            }
        }
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    std::vector<unsigned char> pixels(width * height * 3);
    const auto& cells = planet.getCells();

    for (int y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / (height - 1);
        double lat = Math::PI / 2.0 - v * Math::PI;

        for (int x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / (width - 1);
            double lon = u * 2.0 * Math::PI - Math::PI;

            Vector3 pixelPos = Math::sphericalToCartesian(lat, lon);
            double minDist = std::numeric_limits<double>::max();
            const Cell* closestCell = nullptr;

            for (const auto& cell : cells) {
                double dist = Math::distance(pixelPos, cell.position);
                if (dist < minDist) { minDist = dist; closestCell = &cell; }
            }

            int pixelIdx = (y * width + x) * 3;
            if (closestCell) {
                Color c = {0,0,0};
                if (closestCell->height <= sea_level) {
                    c = {20, 20, 80}; // Ocean
                } else {
                    switch (closestCell->soil) {
                        case SoilType::NONE: c = {100, 100, 100}; break; // Bare rock
                        case SoilType::SAND: c = {240, 230, 140}; break; // Khaki
                        case SoilType::CLAY: c = {180, 100, 50}; break;  // Orange/Red
                        case SoilType::LOAM: c = {80, 40, 20}; break;    // Dark Brown
                    }
                }
                pixels[pixelIdx] = c.r; pixels[pixelIdx+1] = c.g; pixels[pixelIdx+2] = c.b;
            }
        }
    }
    return pixels;
}

std::vector<unsigned char> MapExporter::getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level) {
    std::vector<unsigned char> pixels(width * height * 3);
    const auto& cells = planet.getCells();

    for (int y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / (height - 1);
        double lat = Math::PI / 2.0 - v * Math::PI;

        for (int x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / (width - 1);
            double lon = u * 2.0 * Math::PI - Math::PI;

            Vector3 pixelPos = Math::sphericalToCartesian(lat, lon);
            double minDist = std::numeric_limits<double>::max();
            const Cell* closestCell = nullptr;

            for (const auto& cell : cells) {
                double dist = Math::distance(pixelPos, cell.position);
                if (dist < minDist) { minDist = dist; closestCell = &cell; }
            }

            int pixelIdx = (y * width + x) * 3;
            if (closestCell) {
                unsigned char g = static_cast<unsigned char>(closestCell->height * 255);
                pixels[pixelIdx] = g; pixels[pixelIdx+1] = g; pixels[pixelIdx+2] = g;
            }
        }
    }
    return pixels;
}

} // namespace Ravis
