#include "../include/HydrologySimulator.h"
#include "../include/MathUtils.h"
#include "../include/Noise.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>

namespace Ravis {

HydrologySimulator::HydrologySimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void HydrologySimulator::simulate(const SimulationParameters& params) {
    buildDrainageNetwork(params);
    accumulateFlow(params);
    detectLakes(params);
    applyRiparianEffect(params);
}

// ============================================================================
// Paso 1: Red de drenaje global con Priority-Flood desde el océano
// Garantiza que el árbol de `downstream_id` de todo el planeta llegue al mar sin ciclos.
// ============================================================================
void HydrologySimulator::buildDrainageNetwork(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const size_t n = cells.size();
    if (n == 0) return;

    const float sea = params.effective_sea_level();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    // 1. Identificar el verdadero océano conectado mediante BFS desde la fosa más profunda.
    // Así evitamos que una celda aislada a -5m detrás de una colina costera actúe como falso mar.
    std::vector<uint8_t> is_true_ocean(n, 0);
    size_t deepest_id = 0;
    for (size_t i = 1; i < n; ++i) {
        if (cells[i].elevation < cells[deepest_id].elevation) {
            deepest_id = i;
        }
    }

    if (cells[deepest_id].elevation <= sea) {
        std::queue<size_t> oq;
        oq.push(deepest_id);
        is_true_ocean[deepest_id] = 1;
        while (!oq.empty()) {
            size_t cur = oq.front();
            oq.pop();
            for (size_t nid : cells[cur].neighbors) {
                if (!is_true_ocean[nid] && cells[nid].elevation <= sea) {
                    is_true_ocean[nid] = 1;
                    oq.push(nid);
                }
            }
        }

        // Cualquier celda aislada bajo el nivel del mar que NO esté conectada al océano global
        // (y no haya sido excavada manualmente con el pincel) se sanea a tierra baja (+1.0m sobre el mar)
        // para que Priority-Flood obligue al río a salir hacia el océano real.
        for (size_t i = 0; i < n; ++i) {
            if (cells[i].elevation <= sea && !is_true_ocean[i] && !cells[i].locked) {
                cells[i].elevation = sea + 1.0f;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        cells[i].downstream_id = i;
        cells[i].river_flow = 0.0f;
        cells[i].is_lake = false;
        cells[i].water_surface_elev = cells[i].elevation;
    }

    struct PQNode {
        float water_elev;
        float ground_elev;
        size_t id;
        bool operator>(const PQNode& o) const {
            if (std::abs(water_elev - o.water_elev) > 1e-4f) {
                return water_elev > o.water_elev;
            }
            return ground_elev > o.ground_elev;
        }
    };

    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
    std::vector<uint8_t> visited(n, 0);

    auto routing_elevation = [&](size_t idx) -> float {
        const auto& p = cells[idx].position;
        float px = static_cast<float>(p.x), py = static_cast<float>(p.y), pz = static_cast<float>(p.z);
        float valleys = rw_fbm3d(px * 14.0f, py * 14.0f, pz * 14.0f, 3, u_seed + 313u) * 18.0f
                      + rw_value_noise3d(px * 40.0f, py * 40.0f, pz * 40.0f, u_seed + 317u) * 2.5f;
        return cells[idx].elevation + valleys;
    };

    // Sembrar Priority-Flood ÚNICAMENTE con el océano verdadero conectado
    for (size_t i = 0; i < n; ++i) {
        if (is_true_ocean[i]) {
            visited[i] = 1;
            cells[i].water_surface_elev = cells[i].elevation;
            cells[i].downstream_id = i;
            pq.push({cells[i].elevation, cells[i].elevation, i});
        }
    }

    if (pq.empty()) {
        visited[deepest_id] = 1;
        pq.push({cells[deepest_id].elevation, cells[deepest_id].elevation, deepest_id});
    }

    const float kSlopeEps = 0.02f;
    while (!pq.empty()) {
        PQNode curr = pq.top();
        pq.pop();

        size_t u = curr.id;
        for (size_t v : cells[u].neighbors) {
            if (visited[v]) continue;
            visited[v] = 1;

            cells[v].downstream_id = u;
            float r_elev = routing_elevation(v);
            if (cells[v].elevation <= curr.water_elev) {
                cells[v].water_surface_elev = curr.water_elev + kSlopeEps;
            } else {
                cells[v].water_surface_elev = cells[v].elevation;
            }
            pq.push({cells[v].water_surface_elev, r_elev, v});
        }
    }
}

// ============================================================================
// Paso 2: Cabeceras de Montaña + Acumulación Monótona hacia el Mar
// ============================================================================
void HydrologySimulator::accumulateFlow(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();
    const size_t n = cells.size();
    if (n == 0) return;

    std::vector<size_t> sortedIndices(n);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&cells](size_t a, size_t b) {
            if (cells[a].water_surface_elev != cells[b].water_surface_elev) {
                return cells[a].water_surface_elev > cells[b].water_surface_elev;
            }
            return cells[a].elevation > cells[b].elevation;
        });

    int res_mult = 1 << std::max(0, params.subdivision_level - 5);
    float cell_area_scale = 1.0f / static_cast<float>(res_mult * res_mult);
    float hop_scale = 1.0f / static_cast<float>(res_mult);

    std::vector<float> base_runoff(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        if (cells[i].elevation <= sea) continue;

        float elev_above = cells[i].elevation - sea;
        float max_drop = 0.0f;
        for (size_t nid : cells[i].neighbors) {
            float d = cells[i].elevation - cells[nid].elevation;
            if (d > max_drop) max_drop = d;
        }

        float p = cells[i].precipitation;
        float alt_factor = std::max(0.0f, std::min(2.5f, (elev_above - 600.0f) / 1400.0f));
        float slope_factor = std::max(0.0f, std::min(1.8f, max_drop / (180.0f * hop_scale + 20.0f)));
        float snowmelt_bonus = (elev_above > 1800.0f && cells[i].temperature < 0.28f) ? 2.5f : 0.0f;

        float mountain_multiplier = 0.15f + 4.5f * alt_factor * alt_factor + 2.2f * slope_factor + snowmelt_bonus;

        // En desiertos bajos (< 600m y p < 0.16), la lluvia local se evapora casi por completo in-situ,
        // pero los ríos que YA vienen bajando desde las montañas conservan su caudal hasta el mar.
        float aridity_gate = (elev_above < 600.0f && p < 0.18f) ? (p * p * 0.25f) : p;
        float effective_p = (elev_above > 1100.0f) ? std::max(p, 0.26f) : aridity_gate;

        base_runoff[i] = effective_p * mountain_multiplier * cell_area_scale * 14.0f;
    }

    // 2 pasadas: la primera establece los cauces principales y la segunda atrae lateralmente
    // los escurrimientos de ladera hacia esos cauces, conservando el 100% del caudal aguas abajo
    // para que ningún río se corte antes de llegar al océano.
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < n; ++i) {
            cells[i].river_flow = base_runoff[i];
        }

        for (size_t idx : sortedIndices) {
            if (cells[idx].elevation <= sea) continue;

            float my_water = cells[idx].water_surface_elev;
            float my_elev  = cells[idx].elevation;
            bool in_depression = (my_water > my_elev + 0.05f);

            // Solo redirigir en laderas abiertas (nunca dentro de una depresión resuelta por Priority-Flood,
            // para no romper la ruta de escape de la cuenca hacia el mar).
            if (!in_depression) {
                size_t best_down = cells[idx].downstream_id;
                float best_score = -1.0f;

                for (size_t nid : cells[idx].neighbors) {
                    if (cells[nid].water_surface_elev < my_water && cells[nid].elevation < my_elev) {
                        float drop = my_elev - cells[nid].elevation;
                        float flow_pull = 1.0f + 5.0f * std::sqrt(cells[nid].river_flow);
                        float score = drop * flow_pull;
                        if (score > best_score) {
                            best_score = score;
                            best_down = nid;
                        }
                    }
                }
                cells[idx].downstream_id = best_down;
            }

            size_t downId = cells[idx].downstream_id;
            if (downId != idx && downId < n) {
                // Conservación monótona: el caudal siempre suma hacia aguas abajo, garantizando
                // que todo río que supere el umbral visual llegue ininterrumpidamente al océano.
                cells[downId].river_flow += cells[idx].river_flow;
            }
        }
    }
}

// ============================================================================
// Paso 3: Detección física de lagos a lo largo de los ríos
// ============================================================================
void HydrologySimulator::detectLakes(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const size_t n = cells.size();
    const float sea = params.effective_sea_level();

    int res_mult = 1 << std::max(0, params.subdivision_level - 5);
    const size_t max_lake_cells = std::min<size_t>(n / 1200, static_cast<size_t>(14 * res_mult));
    const size_t max_total_lake_cells = n / 120;
    size_t total_lake_cells = 0;

    std::vector<size_t> candidates;
    for (size_t i = 0; i < n; ++i) {
        if (cells[i].elevation <= sea) continue;
        float depth = cells[i].water_surface_elev - cells[i].elevation;
        if (cells[i].precipitation < 0.14f && cells[i].river_flow < 1.2f) continue;

        if (depth >= 12.0f && depth <= 160.0f && cells[i].river_flow >= 0.30f) {
            bool is_local_min = true;
            for (size_t nid : cells[i].neighbors) {
                if (cells[nid].elevation < cells[i].elevation) {
                    is_local_min = false;
                    break;
                }
            }
            if (is_local_min) candidates.push_back(i);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        return cells[a].river_flow > cells[b].river_flow;
    });

    std::vector<uint8_t> claimed(n, 0);

    for (size_t pit : candidates) {
        if (total_lake_cells >= max_total_lake_cells) break;
        if (claimed[pit]) continue;

        float evap_demand = 0.30f + 1.4f * cells[pit].temperature;
        float max_supported_area = (cells[pit].river_flow / evap_demand) * 4.5f;
        size_t allowed_cells = std::min(max_lake_cells, static_cast<size_t>(std::max(1.0f, max_supported_area)));
        if (allowed_cells < 2) continue;

        float sill_level = std::min(cells[pit].water_surface_elev, cells[pit].elevation + 80.0f);

        std::vector<size_t> basin;
        std::queue<size_t> q;
        q.push(pit);
        claimed[pit] = 1;
        bool touches_ocean = false;
        bool exceeded_cap = false;

        while (!q.empty()) {
            size_t cur = q.front();
            q.pop();
            basin.push_back(cur);

            if (basin.size() > allowed_cells) {
                exceeded_cap = true;
                break;
            }

            for (size_t nid : cells[cur].neighbors) {
                if (cells[nid].elevation <= sea) {
                    touches_ocean = true;
                    continue;
                }
                if (!claimed[nid] && cells[nid].elevation <= sill_level &&
                    cells[nid].water_surface_elev <= cells[pit].water_surface_elev + 2.0f) {
                    claimed[nid] = 1;
                    q.push(nid);
                }
            }
        }

        if (!touches_ocean && !exceeded_cap && basin.size() >= 2) {
            for (size_t c : basin) {
                cells[c].is_lake = true;
            }
            total_lake_cells += basin.size();
        }
    }
}

// ============================================================================
// Paso 4: Efecto ribereño
// ============================================================================
void HydrologySimulator::applyRiparianEffect(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const float sea = params.effective_sea_level();

    std::vector<float> flows;
    flows.reserve(cells.size() / 4);
    for (const auto& cell : cells) {
        if (cell.elevation > sea && cell.river_flow > 0.0f && !cell.is_lake) {
            flows.push_back(cell.river_flow);
        }
    }

    if (flows.empty()) return;
    size_t kth = static_cast<size_t>(flows.size() * 0.96);
    std::nth_element(flows.begin(), flows.begin() + kth, flows.end());
    float riverThreshold = flows[kth];

    std::vector<float> moistureBoost(cells.size(), 0.0f);

    for (size_t i = 0; i < cells.size(); ++i) {
        bool isRiver = (cells[i].river_flow >= riverThreshold && cells[i].elevation > sea && !cells[i].is_lake);
        if (isRiver || cells[i].is_lake) {
            moistureBoost[i] = std::max(moistureBoost[i], 0.25f);
            for (size_t nid : cells[i].neighbors) {
                if (cells[nid].elevation > sea) {
                    moistureBoost[nid] = std::max(moistureBoost[nid], 0.12f);
                }
            }
        }
    }

    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].moisture = std::min(1.0f, cells[i].moisture + moistureBoost[i]);
        cells[i].precipitation = std::min(1.0f, cells[i].precipitation + moistureBoost[i] * 0.40f);
    }
}

} // namespace Ravis