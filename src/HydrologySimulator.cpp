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

    // Step 1.5: Resolve depressions. For every pit, BFS out to an escape (ocean
    // or lower ground), tracking the highest cell crossed — the sill. A pit
    // whose basin is small and shallow becomes a lake (flag only, no elevation
    // change); anything larger/deeper is carved down to the outlet so it drains.
    // Hard caps keep a bad noise field from flooding the whole continent.
    const int    LAKE_MAX_BASIN   = 250;                  // cells per lake
    const float  LAKE_MAX_SILL    = 220.0f;               // m the sill may sit above the pit
    const float  CLIMB_BUDGET     = 400.0f;               // m of climb allowed to find an escape
    const size_t LAKE_MAX_TOTAL   = cells.size() / 120;   // ~0.8% of all cells
    size_t lakeCellsUsed = 0;

    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= params.sea_level) continue;
        if (cells[i].downstream_id != i) continue; // not a pit
        if (cells[i].is_lake) continue;            // already flooded from another pit

        const float pitElev = cells[i].elevation;

        // --- find an escape and the sill along the way ---
        std::queue<size_t> q;
        std::vector<size_t> parent(cells.size(), static_cast<size_t>(-1));
        std::vector<char> vis(cells.size(), 0);
        q.push(i);
        vis[i] = 1;
        size_t outlet = static_cast<size_t>(-1);

        while (!q.empty()) {
            size_t cur = q.front();
            q.pop();
            if (cur != i && (cells[cur].elevation < pitElev ||
                             cells[cur].elevation <= params.sea_level)) {
                outlet = cur;
                break;
            }
            for (size_t nid : cells[cur].neighbors) {
                if (!vis[nid] && cells[nid].elevation < pitElev + CLIMB_BUDGET) {
                    vis[nid] = 1;
                    parent[nid] = cur;
                    q.push(nid);
                }
            }
        }
        if (outlet == static_cast<size_t>(-1)) continue; // fully closed — leave as a pit

        std::vector<size_t> path; // outlet -> ... -> i
        for (size_t n = outlet; n != static_cast<size_t>(-1); n = parent[n]) path.push_back(n);

        float sill = pitElev;
        for (size_t k = 1; k + 1 < path.size(); ++k)
            sill = std::max(sill, cells[path[k]].elevation);

        // --- gather the basin (land cells strictly below the sill), capped ---
        std::vector<size_t> basin;
        bool basinTooBig = false;
        bool basinTouchesOcean = false;
        {
            std::queue<size_t> bq;
            std::vector<char> bvis(cells.size(), 0);
            bq.push(i);
            bvis[i] = 1;
            while (!bq.empty()) {
                size_t cur = bq.front();
                bq.pop();
                basin.push_back(cur);
                if (basin.size() > static_cast<size_t>(LAKE_MAX_BASIN)) { basinTooBig = true; break; }
                for (size_t nid : cells[cur].neighbors) {
                    if (cells[nid].elevation <= params.sea_level) { basinTouchesOcean = true; continue; }
                    if (!bvis[nid] && !cells[nid].is_lake && cells[nid].elevation < sill) {
                        bvis[nid] = 1;
                        bq.push(nid);
                    }
                }
            }
        }

        // Only a genuinely land-locked basin becomes a lake: it must spill over
        // land (not straight into the sea) and no basin cell may border ocean.
        bool makeLake = !basinTooBig && !basinTouchesOcean &&
                        cells[outlet].elevation > params.sea_level &&
                        (sill - pitElev) <= LAKE_MAX_SILL &&
                        (lakeCellsUsed + basin.size()) <= LAKE_MAX_TOTAL;

        if (makeLake) {
            for (size_t c : basin) {
                cells[c].is_lake = true;
                cells[c].downstream_id = i; // pool toward the pit
            }
            lakeCellsUsed += basin.size();
            // Overflow: route pit -> path -> outlet.
            for (size_t k = 0; k + 1 < path.size(); ++k)
                cells[path[k + 1]].downstream_id = path[k];
            cells[i].downstream_id = (path.size() >= 2) ? path[path.size() - 2] : outlet;
        } else {
            // Carve a monotonic channel from the pit down to the outlet.
            float startElev = pitElev;
            float endElev = std::min(cells[outlet].elevation, startElev - 1.0f);
            float step = (startElev - endElev) / static_cast<float>(path.size());
            for (size_t k = 0; k + 1 < path.size(); ++k) {
                size_t cur = path[path.size() - 1 - k]; // from i outward
                size_t nxt = path[path.size() - 2 - k];
                cells[cur].downstream_id = nxt;
                cells[cur].elevation =
                    std::min(cells[cur].elevation, startElev - k * step);
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
    
    // Apply the boost. Riparian zones get more plant-available water (moisture)
    // and, at a smaller rate, count as wetter for biome classification
    // (precipitation) so gallery forests / green lake shores actually form when
    // assignBiomes runs again after hydrology.
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].moisture = std::min(1.0f, cells[i].moisture + moistureBoost[i]);
        cells[i].precipitation = std::min(1.0f, cells[i].precipitation + moistureBoost[i] * 0.6f);
    }
}

} // namespace Ravis
