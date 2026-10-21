#pragma once

#include "GoldbergPolyhedron.h"
#include <cstdint>
#include <string>
#include <vector>

namespace Ravis {

// Pixel -> cell index of an equirectangular map, row-major (y * width + x).
// uint32_t: even subdivision 10 has only ~10.5M cells.
using PixelCellMap = std::vector<uint32_t>;

class MapExporter {
public:
    // Exports the planet data to 2D Equirectangular projection images (PNG).
    static void exportAllMaps(const GoldbergPolyhedron& planet, int width, int height, float sea_level);

    // Precompute the pixel→cell lookup table (O(pixels), run once per mesh)
    // Then pass it to the fast get*Pixels overloads below for O(pixels) rendering.
    static PixelCellMap buildPixelToCellMap(const GoldbergPolyhedron& planet, int width, int height);

    // Fast versions: use a precomputed cellMap for O(pixels) rendering
    static std::vector<unsigned char> getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);
    static std::vector<unsigned char> getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const PixelCellMap& cellMap);

    // Convenience versions: build the cellMap internally (slower if called multiple times)
    static std::vector<unsigned char> getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
};

} // namespace Ravis
