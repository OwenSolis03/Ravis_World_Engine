#include "../include/TectonicSimulator.h"
#include "../include/MathUtils.h"
#include <random>
#include <queue>
#include <iostream>

namespace Ravis {

TectonicSimulator::TectonicSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void TectonicSimulator::generatePlates(int numPlates, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    if (cells.empty()) return;

    std::mt19937 rng(1337);
    std::uniform_int_distribution<size_t> distNode(0, cells.size() - 1);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);

    struct PlateData {
        bool is_oceanic;
        Vector3 rotation_axis;
        float angular_speed;
    };
    std::vector<PlateData> plates(numPlates);

    // 1. Pick seeds and define plate properties
    std::queue<size_t> bfsQueue;
    std::vector<bool> visited(cells.size(), false);

    for (int i = 0; i < numPlates; ++i) {
        // Find an unvisited cell for the seed
        size_t seedId;
        do {
            seedId = distNode(rng);
        } while (visited[seedId]);

        visited[seedId] = true;
        cells[seedId].plate_id = i;
        bfsQueue.push(seedId);

        // 70% chance of oceanic plate (similar to Earth)
        plates[i].is_oceanic = distProb(rng) < 0.70f;
        
        // Random rotation axis
        Vector3 axis(distReal(rng), distReal(rng), distReal(rng));
        plates[i].rotation_axis = Math::normalize(axis);
        // Use params.tectonic_speed instead of hardcoded 0.05f
        plates[i].angular_speed = distProb(rng) * params.tectonic_speed + 0.01f; 
    }

    // 2. Multi-source BFS to expand plates (Spherical Voronoi)
    while (!bfsQueue.empty()) {
        size_t currentId = bfsQueue.front();
        bfsQueue.pop();

        int currentPlate = cells[currentId].plate_id;

        for (size_t neighborId : cells[currentId].neighbors) {
            if (!visited[neighborId]) {
                visited[neighborId] = true;
                cells[neighborId].plate_id = currentPlate;
                bfsQueue.push(neighborId);
            }
        }
    }

    // 3. Assign properties to all cells based on their plate
    for (auto& cell : cells) {
        int pid = cell.plate_id;
        cell.is_oceanic = plates[pid].is_oceanic;
        
        // Assign base elevation
        if (cell.is_oceanic) {
            cell.height = 0.2f + distProb(rng) * 0.05f; // Deep ocean base (~ -4000m)
            cell.bedrock = RockType::BASALT; // Oceanic crust
        } else {
            cell.height = 0.55f + distProb(rng) * 0.05f; // Continental base (~ +500m)
            cell.bedrock = RockType::GRANITE; // Continental crust
        }

        // Calculate velocity vector using cross product: v = w x r
        // w is angular velocity vector (rotation_axis * angular_speed)
        // r is position vector (cell.position)
        Vector3 w(
            plates[pid].rotation_axis.x * plates[pid].angular_speed,
            plates[pid].rotation_axis.y * plates[pid].angular_speed,
            plates[pid].rotation_axis.z * plates[pid].angular_speed
        );
        Vector3 r = cell.position;
        cell.plate_velocity = Vector3(
            w.y * r.z - w.z * r.y,
            w.z * r.x - w.x * r.z,
            w.x * r.y - w.y * r.x
        );
    }
}

void TectonicSimulator::simulateStep(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    std::vector<float> height_deltas(cells.size(), 0.0f);

    for (size_t i = 0; i < cells.size(); ++i) {
        const Cell& cell = cells[i];

        for (size_t neighborId : cell.neighbors) {
            const Cell& neighbor = cells[neighborId];

            // If we are at a plate boundary
            if (cell.plate_id != neighbor.plate_id) {
                // Vector pointing from cell to neighbor
                Vector3 dirToNeighbor = Math::normalize(Vector3(
                    neighbor.position.x - cell.position.x,
                    neighbor.position.y - cell.position.y,
                    neighbor.position.z - cell.position.z
                ));

                // Relative velocity of neighbor to this cell
                Vector3 relativeVel(
                    neighbor.plate_velocity.x - cell.plate_velocity.x,
                    neighbor.plate_velocity.y - cell.plate_velocity.y,
                    neighbor.plate_velocity.z - cell.plate_velocity.z
                );

                // Dot product indicates convergence or divergence
                // If dot < 0, they are moving towards each other (Convergence)
                // If dot > 0, they are moving apart (Divergence)
                double dot = dirToNeighbor.x * relativeVel.x + 
                             dirToNeighbor.y * relativeVel.y + 
                             dirToNeighbor.z * relativeVel.z;

                float force = std::abs(static_cast<float>(dot));

                if (dot < -0.001) { // Convergence
                    if (cell.is_oceanic && neighbor.is_oceanic) {
                        if (cell.plate_id > neighbor.plate_id) {
                            height_deltas[i] += force * 0.5f * params.orogenesis_factor; // Volcanoes
                        } else {
                            height_deltas[i] -= force * 0.8f * params.orogenesis_factor; // Trench
                        }
                    } else if (!cell.is_oceanic && !neighbor.is_oceanic) {
                        // Continent-Continent: massive uplift (Himalayas)
                        height_deltas[i] += force * 1.5f * params.orogenesis_factor;
                    } else {
                        // Oceanic-Continental: Oceanic subducts, Continental uplifts
                        if (cell.is_oceanic) {
                            height_deltas[i] -= force * 1.0f * params.orogenesis_factor; // Trench
                        } else {
                            height_deltas[i] += force * 1.0f * params.orogenesis_factor; // Coastal mountains (Andes)
                        }
                    }
                } else if (dot > 0.001) { // Divergence
                    // Rift valley or mid-ocean ridge. 
                    // Height decreases, magma fills in (creates new oceanic crust over time, but for now just height drop)
                    height_deltas[i] -= force * 0.5f;
                    
                    // Small thermal uplift right at the ridge
                    if (cell.is_oceanic) {
                        height_deltas[i] += force * 0.2f; 
                    }
                }
            }
        }
    }

    // Apply height deltas and smoothing
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].height += height_deltas[i] * 0.1f; // damping factor
        
        // Clamp to valid range
        if (cells[i].height < 0.0f) cells[i].height = 0.0f;
        if (cells[i].height > 1.0f) cells[i].height = 1.0f;
    }
}

void TectonicSimulator::generateHotspots(int numHotspots) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distStrength(0.005f, 0.02f);

    hotspots.clear();
    for (int i = 0; i < numHotspots; ++i) {
        Vector3 pos(distReal(rng), distReal(rng), distReal(rng));
        Hotspot h = { Math::normalize(pos), distStrength(rng) };
        hotspots.push_back(h);
    }
}

void TectonicSimulator::simulateHotspots() {
    auto& cells = planet.getCells();
    
    // For each cell, check distance to hotspots
    for (auto& cell : cells) {
        // Hotspots usually form islands in oceanic crust (e.g., Hawaii)
        // They can also form supervolcanoes in continental crust (e.g., Yellowstone), but we'll focus on oceanic uplift here.
        if (cell.is_oceanic) {
            for (const auto& hotspot : hotspots) {
                double dist = Math::distance(cell.position, hotspot.position);
                if (dist < 0.1) { // Influence radius
                    float intensity = (1.0f - static_cast<float>(dist) / 0.1f);
                    cell.height += hotspot.strength * intensity;
                    
                    if (cell.height > 1.0f) cell.height = 1.0f;
                }
            }
        }
    }
}

void TectonicSimulator::simulate(int iterations, const SimulationParameters& params) {
    generateHotspots(10); // 10 static hotspots

    for (int i = 0; i < iterations; ++i) {
        simulateStep(params);
        simulateHotspots();
        
        // Optional: apply a slight blur/erosion to height map so mountains aren't just 1 cell wide
        if (i % 5 == 0) {
            auto& cells = planet.getCells();
            std::vector<float> smoothed(cells.size());
            for (size_t j = 0; j < cells.size(); ++j) {
                float sum = cells[j].height;
                float count = 1.0f;
                for (size_t neighborId : cells[j].neighbors) {
                    sum += cells[neighborId].height;
                    count += 1.0f;
                }
                smoothed[j] = sum / count;
            }
            for (size_t j = 0; j < cells.size(); ++j) {
                cells[j].height = cells[j].height * 0.8f + smoothed[j] * 0.2f;
            }
        }
    }

    // Metamorphism step: extreme uplift converts rock to METAMORPHIC
    auto& cells = planet.getCells();
    for (auto& cell : cells) {
        if (cell.height > 0.8f) { // Very high mountains experience extreme pressure
            cell.bedrock = RockType::METAMORPHIC;
        }
    }
}

} // namespace Ravis
