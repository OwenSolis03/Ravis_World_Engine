#include "../include/ErosionSimulator.h"
#include "../include/MathUtils.h"
#include <random>
#include <iostream>
#include <omp.h>

namespace Ravis {

ErosionSimulator::ErosionSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void ErosionSimulator::simulateErosion(int numDrops, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // Configurable erosion parameters
    const float dt = 1.0f;           // Time step
    const float min_slope = 0.01f;   // Minimum slope to avoid zero capacity
    const float capacity_factor = 4.0f; // Multiplier for sediment capacity
    const float deposition_rate = 0.1f; // How much sediment drops per step
    const float evaporation_rate = 0.02f; // How much water evaporates
    const float friction = 0.05f;    // Friction reducing velocity
    const int max_lifetime = 30;     // Max steps a droplet can live

    // Use OpenMP to run droplets in parallel
    #pragma omp parallel
    {
        // Thread-local RNG
        std::mt19937 rng(1337 + omp_get_thread_num());
        std::uniform_int_distribution<size_t> distNode(0, cells.size() - 1);
        
        #pragma omp for
        for (int i = 0; i < numDrops; ++i) {
            // Spawn droplet
            size_t currentId = distNode(rng);
            
            // Prefer dropping on land
            while (cells[currentId].height <= params.sea_level) {
                currentId = distNode(rng);
            }

            float water = cells[currentId].moisture * 10.0f; // Scale up for erosion effect
            float velocity = 1.0f;
            float sediment = 0.0f;

            for (int lifetime = 0; lifetime < max_lifetime; ++lifetime) {
                // Find lowest neighbor
                size_t lowestNeighbor = currentId;
                float minHeight = cells[currentId].height;
                
                for (size_t nid : cells[currentId].neighbors) {
                    if (cells[nid].height < minHeight) {
                        minHeight = cells[nid].height;
                        lowestNeighbor = nid;
                    }
                }

                // If in a pit, drop sediment and die
                if (lowestNeighbor == currentId) {
                    #pragma omp atomic
                    cells[currentId].height += sediment;
                    break;
                }
                
                float heightDiff = cells[currentId].height - cells[lowestNeighbor].height;
                float dist = Math::distance(cells[currentId].position, cells[lowestNeighbor].position);
                float slope = heightDiff / dist;

                // Calculate sediment capacity
                float capacity = std::max(slope, min_slope) * velocity * water * capacity_factor;

                // Differential erosion factor based on bedrock
                float base_erosion = 0.1f;
                switch(cells[currentId].bedrock) {
                    case RockType::BASALT: base_erosion = 0.05f; break;
                    case RockType::GRANITE: base_erosion = 0.02f; break;
                    case RockType::SANDSTONE: base_erosion = 0.20f; break;
                    case RockType::SHALE_LIMESTONE: base_erosion = 0.30f; break;
                    case RockType::METAMORPHIC: base_erosion = 0.03f; break;
                }
                float erosion_rate = base_erosion * params.erosion_rate;

                if (sediment > capacity) {
                    // Deposit
                    float amount = (sediment - capacity) * deposition_rate;
                    sediment -= amount;
                    #pragma omp atomic
                    cells[currentId].height += amount;

                    // If a significant amount of sediment is deposited, change bedrock to sedimentary
                    if (amount > 0.001f) {
                        if (cells[currentId].height <= params.sea_level) {
                            cells[currentId].bedrock = RockType::SHALE_LIMESTONE;
                        } else {
                            cells[currentId].bedrock = RockType::SANDSTONE;
                        }
                    }
                } else {
                    // Erode
                    float amount = std::min((capacity - sediment) * erosion_rate, heightDiff);
                    sediment += amount;
                    #pragma omp atomic
                    cells[currentId].height -= amount;
                }

                // Move droplet
                currentId = lowestNeighbor;

                // Update velocity and water
                velocity = std::sqrt(velocity * velocity + heightDiff * 9.8f) * (1.0f - friction);
                water *= (1.0f - evaporation_rate);

                // Stop if we hit the sea
                if (cells[currentId].height <= params.sea_level) {
                    #pragma omp atomic
                    cells[currentId].height += sediment; // Deposit at river delta
                    cells[currentId].bedrock = RockType::SHALE_LIMESTONE;
                    break;
                }
                
                if (water < 0.01f) {
                    #pragma omp atomic
                    cells[currentId].height += sediment;
                    cells[currentId].bedrock = RockType::SANDSTONE;
                    break;
                }
            }
        }
    }
}

} // namespace Ravis
