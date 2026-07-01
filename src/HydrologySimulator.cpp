#include "../include/HydrologySimulator.h"
#include "../include/MathUtils.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <queue>

namespace Ravis {

HydrologySimulator::HydrologySimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void HydrologySimulator::simulate(const SimulationParameters& params) {
    buildDrainageNetwork(params);
    accumulateFlow(params);
    detectLakes(params);
    applyRiparianEffect(params);
}

// ============================================================================
// Step 1: Build drainage network — each land cell points to its lowest neighbor
// ============================================================================
void HydrologySimulator::buildDrainageNetwork(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].downstream_id = i; // Default: points to self (pit)
        cells[i].river_flow = 0.0f;
        cells[i].is_lake = false;
        
        // Only process land cells
        if (cells[i].elevation <= params.sea_level) continue;
        
        // Find the neighbor with steepest descent
        float minElev = cells[i].elevation;
        size_t lowestId = i;
        
        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].elevation < minElev) {
                minElev = cells[nid].elevation;
                lowestId = nid;
            }
        }
        
        cells[i].downstream_id = lowestId;
    }

    // Step 1.5: Carve canyons from pits to the sea (BFS)
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= params.sea_level) continue;
        
        // If it's a pit, run BFS
        if (cells[i].downstream_id == i) {
            std::queue<size_t> q;
            std::vector<size_t> parent(cells.size(), static_cast<size_t>(-1));
            std::vector<bool> visited(cells.size(), false);
            
            q.push(i);
            visited[i] = true;
            
            size_t targetId = static_cast<size_t>(-1);
            
            while (!q.empty()) {
                size_t curr = q.front();
                q.pop();
                
                if (cells[curr].elevation < cells[i].elevation || cells[curr].elevation <= params.sea_level) {
                    targetId = curr;
                    break;
                }
                
                // Allow climbing slightly (to escape the basin), but not climbing Everest
                for (size_t nid : cells[curr].neighbors) {
                    if (!visited[nid] && cells[nid].elevation < cells[i].elevation + 1500.0f) {
                        visited[nid] = true;
                        parent[nid] = curr;
                        q.push(nid);
                    }
                }
            }
            
            // If we found a valid lower ground or ocean
            if (targetId != static_cast<size_t>(-1)) {
                size_t pathNode = targetId;
                std::vector<size_t> path;
                while (pathNode != i) {
                    path.push_back(pathNode);
                    pathNode = parent[pathNode];
                }
                path.push_back(i);
                
                float startElev = cells[i].elevation;
                float endElev = cells[targetId].elevation;
                if (endElev >= startElev) endElev = startElev - 1.0f; 
                
                float elevStep = (startElev - endElev) / static_cast<float>(path.size());
                
                for (size_t p = 0; p < path.size() - 1; ++p) {
                    size_t curr = path[path.size() - 1 - p];
                    size_t next = path[path.size() - 2 - p];
                    
                    cells[curr].downstream_id = next;
                    // Carve the canyon so water physically flows down
                    cells[curr].elevation = std::min(cells[curr].elevation, startElev - p * elevStep);
                }
                cells[targetId].elevation = std::min(cells[targetId].elevation, startElev - (path.size() - 1) * elevStep);
            }
        }
    }
}

// ============================================================================
// Step 2: Flow accumulation — sort cells by elevation (high→low), pass flow downhill
// ============================================================================
void HydrologySimulator::accumulateFlow(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // Create sorted indices (highest elevation first)
    std::vector<size_t> sortedIndices(cells.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&cells](size_t a, size_t b) { return cells[a].elevation > cells[b].elevation; });
    
    // Initialize flow with each cell's precipitation
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation > params.sea_level) {
            cells[i].river_flow = cells[i].precipitation;
        } else {
            cells[i].river_flow = 0.0f;
        }
    }
    
    // Flow accumulation: from high to low, pass flow downstream
    for (size_t idx : sortedIndices) {
        if (cells[idx].elevation <= params.sea_level) continue;
        
        size_t downId = cells[idx].downstream_id;
        if (downId != idx) { // Not a pit
            cells[downId].river_flow += cells[idx].river_flow;
        }
    }
}

// ============================================================================
// Step 3: Detect lakes — pit cells with significant accumulated flow
// ============================================================================
void HydrologySimulator::detectLakes(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // A "pit" is a land cell whose downstream is itself (no lower neighbor)
    // If significant water flows into it, it becomes a lake
    const float lake_threshold = 0.1f; // Minimum accumulated flow to form a lake
    
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= params.sea_level) continue;
        
        // Is this a pit? (flows to itself)
        if (cells[i].downstream_id == i && cells[i].river_flow > lake_threshold) {
            // Mark this cell and nearby low-elevation cells as lake
            cells[i].is_lake = true;
            
            // Flood-fill: expand the lake to include neighbors that are lower than
            // the pit + a small water level rise
            float waterLevel = cells[i].elevation + 50.0f; // 50m water depth
            
            std::queue<size_t> flood;
            flood.push(i);
            
            while (!flood.empty()) {
                size_t current = flood.front();
                flood.pop();
                
                for (size_t nid : cells[current].neighbors) {
                    if (!cells[nid].is_lake && 
                        cells[nid].elevation > params.sea_level &&
                        cells[nid].elevation <= waterLevel) {
                        cells[nid].is_lake = true;
                        cells[nid].river_flow = cells[i].river_flow; // Share the flow
                        flood.push(nid);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Step 4: Riparian effect — rivers and lakes boost moisture of adjacent cells
// ============================================================================
void HydrologySimulator::applyRiparianEffect(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // Determine river threshold dynamically (top 5% of flow values)
    std::vector<float> flows;
    flows.reserve(cells.size());
    for (const auto& cell : cells) {
        if (cell.elevation > params.sea_level && cell.river_flow > 0.0f) {
            flows.push_back(cell.river_flow);
        }
    }
    
    if (flows.empty()) return;
    
    std::sort(flows.begin(), flows.end());
    float riverThreshold = flows[static_cast<size_t>(flows.size() * 0.85)]; // Top 15%
    
    // Boost moisture for cells adjacent to rivers and lakes
    std::vector<float> moistureBoost(cells.size(), 0.0f);
    
    for (size_t i = 0; i < cells.size(); ++i) {
        bool isRiver = (cells[i].river_flow >= riverThreshold && 
                        cells[i].elevation > params.sea_level &&
                        !cells[i].is_lake);
        
        if (isRiver || cells[i].is_lake) {
            // Boost self
            moistureBoost[i] = std::max(moistureBoost[i], 0.3f);
            
            // Boost neighbors (riparian zone)
            for (size_t nid : cells[i].neighbors) {
                if (cells[nid].elevation > params.sea_level) {
                    moistureBoost[nid] = std::max(moistureBoost[nid], 0.15f);
                }
            }
        }
    }
    
    // Apply the boost
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].moisture = std::min(1.0f, cells[i].moisture + moistureBoost[i]);
    }
}

} // namespace Ravis
