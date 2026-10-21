#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Ravis {

// Vector 3D para posiciones y velocidades
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
// Clasificación bioclimática — Fuente única de verdad (Holdridge / Whittaker PET)
// ---------------------------------------------------------------------------
// Usa temperatura media anual (°C), precipitación normalizada (0..1 -> 0..4000 mm/año)
// y acopla la Evapotranspiración Potencial (PET = 58.93 * T_bio en mm/año) para que
// el umbral de aridez escale físicamente con el calor.
inline BiomeType classifyBiome(float tempC, float precip01, float tempSeasonalityC = 0.0f) {
    const float precip_mm = std::max(0.0f, precip01) * 4000.0f; // 1.0 norm = 4000 mm/yr

    // Criosfera y zonas polares/alpinas
    if (tempC < -8.0f) return BiomeType::ICE;
    if (tempC < -1.5f) return BiomeType::TUNDRA;

    // Biotemperatura de Holdridge acotada para índice de aridez
    const float t_bio = std::max(1.0f, std::min(30.0f, tempC));
    const float pet_mm = 52.0f * t_bio + 120.0f;
    const float humidity_index = precip_mm / pet_mm;

    // Zona Boreal / Subártica (-1.5 °C a 5.5 °C)
    if (tempC < 5.5f) {
        if (precip_mm < 150.0f || humidity_index < 0.32f) return BiomeType::TUNDRA;
        if (humidity_index < 0.48f) return BiomeType::STEPPE;
        return BiomeType::BOREAL_FOREST;
    }

    // Zona Templada Fría (5.5 °C a 13.5 °C)
    if (tempC < 13.5f) {
        if (humidity_index < 0.15f || precip_mm < 130.0f) return BiomeType::DESERT;
        if (humidity_index < 0.36f) return BiomeType::STEPPE;
        if (humidity_index < 0.65f) return BiomeType::GRASSLAND;
        if (precip_mm < 1650.0f)    return BiomeType::TEMPERATE_FOREST;
        return BiomeType::TEMPERATE_RAINFOREST;
    }

    // Zona Templada Cálida / Subtropical (13.5 °C a 21.0 °C)
    if (tempC < 21.0f) {
        if (humidity_index < 0.16f || precip_mm < 180.0f) return BiomeType::DESERT;
        if (humidity_index < 0.34f) return BiomeType::STEPPE;
        if (humidity_index < 0.58f) {
            return (tempSeasonalityC > 13.0f) ? BiomeType::GRASSLAND : BiomeType::MEDITERRANEAN;
        }
        if (precip_mm < 1750.0f) return BiomeType::TEMPERATE_FOREST;
        return BiomeType::TEMPERATE_RAINFOREST;
    }

    // Zona Tropical (>= 21.0 °C)
    if (humidity_index < 0.15f || precip_mm < 210.0f) return BiomeType::DESERT;
    if (humidity_index < 0.30f) return BiomeType::THORN_SCRUB;
    if (humidity_index < 0.62f) return BiomeType::SAVANNA;
    if (humidity_index < 1.05f || precip_mm < 1650.0f) return BiomeType::TROPICAL_DRY_FOREST;
    return BiomeType::RAINFOREST;
}

// Colores canónicos de biomas terrestres (RGB lineal 0..1)
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
    default:                              r = 0.12f; g = 0.22f; bl = 0.48f; break;
    }
}

// Representa una celda (hexágono o pentágono dual) en la superficie planetaria
struct Cell {
    size_t id = 0;
    double latitude = 0.0;
    double longitude = 0.0;
    Vector3 position = Vector3(0, 0, 0); // Coordenadas cartesianas 3D en esfera unitaria

    // Topografía y Tectónica
    float elevation = 0.0f;          // Elevación en metros (SI). Negativo = bajo el nivel del mar
    bool is_oceanic = false;         // Tipo de corteza: true = oceánica, false = continental
    int plate_id = -1;
    Vector3 plate_velocity = Vector3(0, 0, 0);
    bool locked = false;             // true si fue editada por el usuario (protege contra erosión/zanjas)

    // Clima y Océano
    float temperature = 0.0f;        // Normalizada 0.0 (-15°C) a 1.0 (+40°C)
    float temp_seasonality = 0.0f;   // Amplitud térmica anual en °C (Verano - Invierno)
    float moisture = 0.0f;           // Humedad atmosférica en columna (0.0 a 1.0)
    float precipitation = 0.0f;      // Precipitación anual normalizada (1.0 = 4000 mm/año)
    Vector3 wind_velocity = Vector3(0, 0, 0);  // Vector velocidad del viento superficial (con magnitud real)
    Vector3 ocean_current = Vector3(0, 0, 0);  // Vector velocidad de corriente marina superficial
    float ocean_temp_anomaly = 0.0f; // Anomalía térmica por corrientes cálidas/frías

    // Geología y Pedología
    RockType bedrock = RockType::BASALT;
    SoilType soil = SoilType::NONE;
    BiomeType biome = BiomeType::OCEAN;
	float vegetation_cover = 0.0f;   // 0.0 (roca/arena/hielo) a 1.0 (dosel forestal cerrado)

    // Geofísica
    float crustal_age = 0.0f;        // Ma — edad de la corteza (gobierna subsidencia y subducción)
    float crustal_thickness = 35.0f; // km — continental ~35-70 km, oceánica ~7 km
    float sediment_depth = 0.0f;     // m — espesor de sedimento depositado por erosión/ríos

    // Hidrología
    bool is_lake = false;            // true si pertenece a un lago o mar interior endorreico
    float water_surface_elev = 0.0f; // Nivel superficial del agua (para lagos sin excavar el terreno base)
    float river_flow = 0.0f;         // Caudal acumulado aguas arriba
    size_t downstream_id = 0;        // ID de la celda hacia la que drena

    // Topología (ordenada en sentido antihorario alrededor de `position`)
    std::vector<size_t> neighbors;
};

} // namespace Ravis