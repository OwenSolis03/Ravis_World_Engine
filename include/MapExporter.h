#pragma once

#include "GoldbergPolyhedron.h"
#include <string>

namespace Ravis {

class MapExporter {
public:
    // Exports the planet data to 2D Equirectangular projection images (PNG).
    static void exportAllMaps(const GoldbergPolyhedron& planet, int width, int height, float sea_level);

    static std::vector<unsigned char> getBiomePixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    
    static std::vector<unsigned char> getLithologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getPedologyPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
    static std::vector<unsigned char> getHeightPixels(const GoldbergPolyhedron& planet, int width, int height, float sea_level);
};

} // namespace Ravis
