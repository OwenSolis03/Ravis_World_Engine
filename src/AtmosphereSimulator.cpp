#include "../include/AtmosphereSimulator.h"
#include "../include/MathUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace Ravis {

AtmosphereSimulator::AtmosphereSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void AtmosphereSimulator::calculatePrimaryClimate(const SimulationParameters& params) {
    calculateTemperatures(params);
    auto& cells = planet.getCells();
    for (auto& cell : cells) {
        // Basic initial moisture based on latitude (Equator is wet, poles are dry)
        float latFactor = std::cos(cell.latitude); 
        cell.moisture = latFactor * 0.5f; 
        if (cell.height <= params.sea_level) {
            cell.moisture = 1.0f;
        }
    }
}

void AtmosphereSimulator::calculateFullClimate(int moistureIterations, const SimulationParameters& params) {
    calculateWinds();
    simulateMoisture(moistureIterations, params);
    assignBiomes();
}

void AtmosphereSimulator::calculateTemperatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    for (auto& cell : cells) {
        // Base temperature based on latitude (Equator = Hot, Poles = Cold)
        float latFactor = std::cos(cell.latitude); 
        float baseTemp = latFactor; 
        
        // Elevation lapse rate: decrease temp by elevation above sea level
        float elevation = cell.height - params.sea_level;
        if (elevation < 0) elevation = 0; // Water buffers temperature
        
        float lapseRate = elevation * 1.5f; 
        
        cell.temperature = baseTemp - lapseRate + params.temp_offset;
        
        if (cell.temperature < 0.0f) cell.temperature = 0.0f;
        if (cell.temperature > 1.0f) cell.temperature = 1.0f;
    }
}

void AtmosphereSimulator::calculateWinds() {
    auto& cells = planet.getCells();
    
    for (auto& cell : cells) {
        double latDeg = cell.latitude * 180.0 / Math::PI;
        
        // Calculate basis vectors tangent to the surface
        Vector3 northPole(0, 0, 1);
        Vector3 eastDir(
            northPole.y * cell.position.z - northPole.z * cell.position.y,
            northPole.z * cell.position.x - northPole.x * cell.position.z,
            northPole.x * cell.position.y - northPole.y * cell.position.x
        );
        eastDir = Math::normalize(eastDir);
        
        Vector3 northDir(
            cell.position.y * eastDir.z - cell.position.z * eastDir.y,
            cell.position.z * eastDir.x - cell.position.x * eastDir.z,
            cell.position.x * eastDir.y - cell.position.y * eastDir.x
        );
        northDir = Math::normalize(northDir);

        Vector3 wind(0,0,0);
        float absLat = std::abs(latDeg);

        // Global Circulation Model (Coriolis effect)
        if (absLat < 30.0) {
            // Hadley Cell: Trade Winds (Easterlies)
            float equatorward = latDeg > 0 ? -1.0f : 1.0f;
            wind = Vector3(-eastDir.x + northDir.x * equatorward * 0.5f,
                           -eastDir.y + northDir.y * equatorward * 0.5f,
                           -eastDir.z + northDir.z * equatorward * 0.5f);
        } else if (absLat < 60.0) {
            // Ferrel Cell: Westerlies
            float poleward = latDeg > 0 ? 1.0f : -1.0f;
            wind = Vector3(eastDir.x + northDir.x * poleward * 0.5f,
                           eastDir.y + northDir.y * poleward * 0.5f,
                           eastDir.z + northDir.z * poleward * 0.5f);
        } else {
            // Polar Cell: Polar Easterlies
            float equatorward = latDeg > 0 ? -1.0f : 1.0f;
            wind = Vector3(-eastDir.x + northDir.x * equatorward * 0.5f,
                           -eastDir.y + northDir.y * equatorward * 0.5f,
                           -eastDir.z + northDir.z * equatorward * 0.5f);
        }
        
        cell.wind_velocity = Math::normalize(wind);
    }
}

void AtmosphereSimulator::simulateMoisture(int iterations, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    for (auto& cell : cells) {
        cell.moisture = 0.0f;
        cell.precipitation = 0.0f;
    }

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<float> next_moisture(cells.size(), 0.0f);
        std::vector<float> rain(cells.size(), 0.0f);

        for (size_t i = 0; i < cells.size(); ++i) {
            auto& cell = cells[i];
            
            float currentMoisture = cell.moisture;
            // 1. Evaporation over oceans (warm water evaporates more)
            if (cell.height <= params.sea_level) {
                currentMoisture += cell.temperature * 0.1f; 
            }
            if (currentMoisture > 1.0f) currentMoisture = 1.0f;
            
            // 2. Find downstream cell
            size_t downstreamId = i;
            double maxDot = -1.0;
            
            for (size_t nid : cell.neighbors) {
                const auto& neighbor = cells[nid];
                Vector3 dir = Math::normalize(Vector3(
                    neighbor.position.x - cell.position.x,
                    neighbor.position.y - cell.position.y,
                    neighbor.position.z - cell.position.z
                ));
                
                double dot = dir.x * cell.wind_velocity.x + 
                             dir.y * cell.wind_velocity.y + 
                             dir.z * cell.wind_velocity.z;
                if (dot > maxDot) {
                    maxDot = dot;
                    downstreamId = nid;
                }
            }
            
            // 3. Orographic Precipitation
            if (downstreamId != i) {
                const auto& downstream = cells[downstreamId];
                float heightDiff = downstream.height - cell.height;
                
                float rainAmount = 0.0f;
                if (heightDiff > 0.02f) { // Hit a mountain
                    rainAmount = currentMoisture * 0.8f; 
                } else if (currentMoisture > 0.8f) { // Saturated
                    rainAmount = currentMoisture * 0.2f;
                }
                
                rain[i] += rainAmount;
                next_moisture[downstreamId] += (currentMoisture - rainAmount);
            } else {
                next_moisture[i] += currentMoisture;
            }
        }
        
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i].moisture = std::min(1.0f, next_moisture[i]);
            cells[i].precipitation += rain[i];
        }
    }
    
    // Normalize precipitation for rendering
    float maxPrecip = 0.01f;
    for (const auto& cell : cells) {
        if (cell.precipitation > maxPrecip) maxPrecip = cell.precipitation;
    }
    for (auto& cell : cells) {
        cell.precipitation /= maxPrecip;        
    }
}

void AtmosphereSimulator::assignBiomes() {
    auto& cells = planet.getCells();
    for (auto& cell : cells) {
        if (cell.height <= 0.45f) { 
            cell.biome = BiomeType::OCEAN;
            continue;
        }

        if (cell.temperature < 0.2f) {
            cell.biome = BiomeType::ICE;
        } else if (cell.temperature < 0.4f) {
            if (cell.precipitation < 0.3f) cell.biome = BiomeType::TUNDRA;
            else cell.biome = BiomeType::BOREAL_FOREST;
        } else if (cell.temperature < 0.7f) {
            if (cell.precipitation < 0.2f) cell.biome = BiomeType::DESERT;
            else if (cell.precipitation < 0.5f) cell.biome = BiomeType::GRASSLAND;
            else cell.biome = BiomeType::TEMPERATE_FOREST;
        } else {
            if (cell.precipitation < 0.2f) cell.biome = BiomeType::DESERT;
            else if (cell.precipitation < 0.5f) cell.biome = BiomeType::SAVANNA;
            else cell.biome = BiomeType::RAINFOREST;
        }
    }
}

} // namespace Ravis
