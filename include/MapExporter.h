#pragma once

#include "GoldbergPolyhedron.h"
#include <string>
#include <vector>

namespace Ravis {

class MapExporter {
public:
    // Exports the planet data to 2D Equirectangular projection images (PNG).
    static void exportAllMaps(const GoldbergPolyhedron& planet, int width, int height, float sea_level);

    // Precompute the pixel→cell lookup table (O(pixels * cells) — run ONCE)
    // Then pass it to the fast get*Pixels overloads below for O(pixels) rendering.
    static std::vector<size_t> buildPixelToCellMap(const GoldbergPolyhedron& planet, int width, int height);

    // Fast versions: use a precomputed cellMap for O(pixels) rendering
    static std::vector<unsigned char> getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getHydrologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getTemperaturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);
    static std::vector<unsigned char> getMoisturePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level, const std::vector<size_t>& cellMap);

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
