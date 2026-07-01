#define _USE_MATH_DEFINES
#include "../include/AtmosphereSimulator.h"
#include "../include/MathUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <queue>

#include <cuda_runtime.h>

namespace Ravis {

// CPU-side noise for climate distortion
static float cpu_hash_noise3d(float x, float y, float z) {
    float d1 = x * 127.1f + y * 311.7f + z * 74.7f;
    float d2 = x * 269.5f + y * 183.3f + z * 246.1f;
    float d3 = x * 419.2f + y * 371.9f + z * 128.9f;
    
    float h1 = std::sin(d1) * 43758.5453f;
    float h2 = std::sin(d2) * 22578.1459f;
    float h3 = std::sin(d3) * 10003.2987f;
    
    h1 = h1 - std::floor(h1);
    h2 = h2 - std::floor(h2);
    h3 = h3 - std::floor(h3);
    
    return std::sin(h1 * 6.2832f + h2 * 3.1416f + h3 * 1.5708f);
}

static float cpu_hash_fbm3d(float x, float y, float z, int octaves) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float total_amp = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        value += cpu_hash_noise3d(x * frequency, y * frequency, z * frequency) * amplitude;
        total_amp += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value / total_amp;
}

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void wind_update_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* lat,
    float* wind_u, float* wind_v, float* wind_w,
    const float* pressure,
    const int* neighbors, const int* num_neighbors,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float p = pressure[i];

    // Compute pressure gradient
    float grad_x = 0, grad_y = 0, grad_z = 0;
    int nn = num_neighbors[i];
    for(int n=0; n<nn; ++n) {
        int ni = neighbors[i * 6 + n];
        float nx = pos_x[ni], ny = pos_y[ni], nz = pos_z[ni];
        float np = pressure[ni];
        
        float dx = nx - px;
        float dy = ny - py;
        float dz = nz - pz;
        float dist_sq = dx*dx + dy*dy + dz*dz + 1e-6f;
        
        float grad_mag = (np - p) / dist_sq;
        grad_x += dx * grad_mag;
        grad_y += dy * grad_mag;
        grad_z += dz * grad_mag;
    }

    // Coriolis effect: f = 2 * Omega * sin(lat)
    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];
    float sin_lat = sinf(lat[i]);
    float f = 2.0f * 1.5f * sin_lat; // Tweaked Omega for pronounced banding

    float cor_x = f * (py * ww - pz * wv);
    float cor_y = f * (pz * wu - px * ww);
    float cor_z = f * (px * wv - py * wu);

    // Update wind
    wu = wu * 0.98f - grad_x * dt + cor_x * dt;
    wv = wv * 0.98f - grad_y * dt + cor_y * dt;
    ww = ww * 0.98f - grad_z * dt + cor_z * dt;

    // Project back to tangent plane
    float dot = wu * px + wv * py + ww * pz;
    wu -= dot * px;
    wv -= dot * py;
    ww -= dot * pz;

    // Normalize/Cap
    float speed = sqrtf(wu*wu + wv*wv + ww*ww);
    if (speed > 1.0f) {
        wu /= speed; wv /= speed; ww /= speed;
    }

    wind_u[i] = wu;
    wind_v[i] = wv;
    wind_w[i] = ww;
}

__global__ void pressure_update_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* wind_u, const float* wind_v, const float* wind_w,
    float* pressure, float* next_pressure,
    const int* neighbors, const int* num_neighbors,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];
    
    float divergence = 0.0f;
    int nn = num_neighbors[i];
    for(int n=0; n<nn; ++n) {
        int ni = neighbors[i * 6 + n];
        float nx = pos_x[ni], ny = pos_y[ni], nz = pos_z[ni];
        float nwu = wind_u[ni], nwv = wind_v[ni], nww = wind_w[ni];
        
        float dx = nx - px; float dy = ny - py; float dz = nz - pz;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        
        // Vector towards neighbor
        float vx = dx/dist; float vy = dy/dist; float vz = dz/dist;
        
        // Outward wind component
        float out_wind = wu * vx + wv * vy + ww * vz;
        float in_wind = -(nwu * vx + nwv * vy + nww * vz);
        
        divergence += out_wind - in_wind;
    }
    
    next_pressure[i] = pressure[i] - divergence * dt * 0.1f;
}

AtmosphereSimulator::AtmosphereSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void AtmosphereSimulator::calculatePrimaryClimate(const SimulationParameters& params) {
    calculateTemperatures(params);
    auto& cells = planet.getCells();
    for (auto& cell : cells) {
        // Distort latitude with low-frequency noise for precipitation
        float noise = cpu_hash_fbm3d(cell.position.x * 2.0f, cell.position.y * 2.0f, cell.position.z * 2.0f, 3);
        float distorted_lat = cell.latitude + noise * 0.3f; 
        float lat_deg = std::abs(distorted_lat) * 180.0f / 3.14159265359f;
        
        float moisture_base = 0.0f;
        if (lat_deg < 30.0f) {
            // Equator (Rainforests) down to 30° (Deserts)
            moisture_base = 1.0f - (lat_deg / 30.0f) * 0.9f;
        } else if (lat_deg < 60.0f) {
            // 30° (Deserts) up to 60° (Temperate forests)
            moisture_base = 0.1f + ((lat_deg - 30.0f) / 30.0f) * 0.5f;
        } else {
            // 60° down to 90° (Tundra/Ice)
            moisture_base = 0.6f - ((lat_deg - 60.0f) / 30.0f) * 0.4f;
        }
        
        moisture_base = std::max(0.0f, std::min(1.0f, moisture_base));
        
        cell.moisture = moisture_base;
        cell.precipitation = moisture_base; // Base precipitation before wind simulation
        
        if (cell.elevation <= params.effective_sea_level()) {
            cell.moisture = 1.0f;
            cell.precipitation = 1.0f;
        }
    }
}

void AtmosphereSimulator::calculateFullClimate(int moistureIterations, const SimulationParameters& params) {
    calculateWinds(params);
    simulateMoisture(moistureIterations, params);
    assignBiomes(params);
}

void AtmosphereSimulator::calculateTemperatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // Convert LGM anomaly from °C to normalized scale (55°C range maps to 0-1)
    float lgm_offset = params.lgm_temp_anomaly / 55.0f;
    float effective_sea = params.effective_sea_level();
    
    // 23° axial tilt to shift the thermal equator
    float axial_tilt = 23.0f * 3.14159265359f / 180.0f; 
    
    for (auto& cell : cells) {
        // Add low-frequency FBM noise to undulate the isotherms organically
        float noise = cpu_hash_fbm3d(cell.position.x * 2.5f, cell.position.y * 2.5f, cell.position.z * 2.5f, 3);
        
        // Calculate thermal latitude (apply tilt and noise)
        float thermal_lat = cell.latitude + axial_tilt * 0.5f + noise * 0.2f; 
        // Clamp to poles
        thermal_lat = std::max(-3.14159265359f/2.0f, std::min(3.14159265359f/2.0f, thermal_lat));
        
        float latFactor = std::cos(thermal_lat);
        float baseTemp = latFactor; 
        
        float elevAboveSea = cell.elevation - effective_sea;
        if (elevAboveSea < 0) elevAboveSea = 0;
        
        float lapseRate = elevAboveSea * 0.00012f; // Mountains are colder
        
        cell.temperature = baseTemp - lapseRate + params.temp_offset + lgm_offset;
        
        // Oceans have higher heat capacity (moderated temperatures)
        if (cell.elevation <= effective_sea) {
            cell.temperature = cell.temperature * 0.7f + 0.15f; 
        }
        
        if (cell.temperature < 0.0f) cell.temperature = 0.0f;
        if (cell.temperature > 1.0f) cell.temperature = 1.0f;
    }
}

void AtmosphereSimulator::calculateWinds(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    int num_cells = cells.size();

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_lat(num_cells), h_pressure(num_cells);
    std::vector<float> h_wind_u(num_cells, 0), h_wind_v(num_cells, 0), h_wind_w(num_cells, 0);
    std::vector<int> h_neighbors(num_cells * 6, -1);
    std::vector<int> h_num_neighbors(num_cells, 0);

    for (int i = 0; i < num_cells; ++i) {
        h_pos_x[i] = cells[i].position.x;
        h_pos_y[i] = cells[i].position.y;
        h_pos_z[i] = cells[i].position.z;
        h_lat[i] = cells[i].latitude;
        
        // Initial pressure based on temperature (hot air rises -> low surface pressure)
        // Cold air sinks -> high surface pressure
        h_pressure[i] = 1.0f - cells[i].temperature; 

        h_num_neighbors[i] = cells[i].neighbors.size();
        for (size_t n = 0; n < cells[i].neighbors.size(); ++n) {
            h_neighbors[i * 6 + n] = cells[i].neighbors[n];
        }
    }

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_lat, *d_pressure, *d_next_pressure;
    float *d_wind_u, *d_wind_v, *d_wind_w;
    int *d_neighbors, *d_num_neighbors;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lat, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pressure, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_pressure, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lat, h_lat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pressure, h_pressure.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_u, h_wind_u.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_v, h_wind_v.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_w, h_wind_w.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    // Simulate SWE for swe_iterations to reach steady state
    float dt = 0.05f;
    for (int iter = 0; iter < params.swe_iterations; ++iter) {
        wind_update_kernel<<<numBlocks, blockSize>>>(num_cells, d_pos_x, d_pos_y, d_pos_z, d_lat,
                                                    d_wind_u, d_wind_v, d_wind_w, d_pressure,
                                                    d_neighbors, d_num_neighbors, dt);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        pressure_update_kernel<<<numBlocks, blockSize>>>(num_cells, d_pos_x, d_pos_y, d_pos_z,
                                                        d_wind_u, d_wind_v, d_wind_w,
                                                        d_pressure, d_next_pressure,
                                                        d_neighbors, d_num_neighbors, dt);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        std::swap(d_pressure, d_next_pressure);
    }

    CHECK_CUDA(cudaMemcpy(h_wind_u.data(), d_wind_u, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wind_v.data(), d_wind_v, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wind_w.data(), d_wind_w, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cells; ++i) {
        cells[i].wind_velocity = Math::normalize(Vector3(h_wind_u[i], h_wind_v[i], h_wind_w[i]));
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_lat); cudaFree(d_pressure); cudaFree(d_next_pressure);
    cudaFree(d_wind_u); cudaFree(d_wind_v); cudaFree(d_wind_w);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
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
            if (cell.elevation <= params.effective_sea_level()) {
                currentMoisture += cell.temperature * 0.1f; 
            }
            if (currentMoisture > 1.0f) currentMoisture = 1.0f;
            
            struct NeighborWind { size_t id; float dot; float elevDiff; };
            std::vector<NeighborWind> downwind;
            float totalDot = 0.0f;
            
            for (size_t nid : cell.neighbors) {
                const auto& neighbor = cells[nid];
                Vector3 dir = Math::normalize(Vector3(
                    neighbor.position.x - cell.position.x,
                    neighbor.position.y - cell.position.y,
                    neighbor.position.z - cell.position.z
                ));
                
                float dot = static_cast<float>(
                    dir.x * cell.wind_velocity.x + 
                    dir.y * cell.wind_velocity.y + 
                    dir.z * cell.wind_velocity.z);
                
                if (dot > 0.0f) {
                    float elevDiff = neighbor.elevation - cell.elevation;
                    downwind.push_back({nid, dot, elevDiff});
                    totalDot += dot;
                }
            }
            
            float maxUpslope = 0.0f;
            for (const auto& nw : downwind) {
                if (nw.elevDiff > maxUpslope) maxUpslope = nw.elevDiff;
            }
            
            float rainAmount = 0.0f;
            if (maxUpslope > 0.0f) {
                float rainFraction = std::min(1.0f, maxUpslope / 1000.0f) * 0.6f;
                rainAmount = currentMoisture * rainFraction;
            } 
            
            // Baseline precipitation (evaporation/rain cycle on flat plains)
            float baseline = currentMoisture * 0.05f; 
            if (currentMoisture > 0.6f) baseline += currentMoisture * 0.1f;
            if (rainAmount < baseline) rainAmount = baseline;
            
            rain[i] += rainAmount;
            float remaining = currentMoisture - rainAmount;
            
            if (totalDot > 0.0f && remaining > 0.0f) {
                for (const auto& nw : downwind) {
                    float weight = nw.dot / totalDot;
                    next_moisture[nw.id] += remaining * weight;
                }
            } else {
                next_moisture[i] += remaining;
            }
        }
        
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i].moisture = std::min(1.0f, next_moisture[i]);
            cells[i].precipitation += rain[i];
        }
    }
    
    // Normalize precipitation by a fixed "reasonable maximum" rather than the absolute global maximum.
    // This prevents a single outlier mountain cell from destroying the world's biomes.
    for (auto& cell : cells) {
        cell.precipitation = std::min(1.0f, cell.precipitation / 3.0f);        
    }
}

void AtmosphereSimulator::assignBiomes(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    
    // Step 1: Assign Ocean and compute ideal biomes for ALL cells temporarily
    std::vector<BiomeType> ideal_biomes(cells.size(), BiomeType::OCEAN);
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= params.effective_sea_level()) { 
            cells[i].biome = BiomeType::OCEAN;
            continue;
        }

        float temp_c = cells[i].temperature * 55.0f - 15.0f;
        float biotemp = std::max(0.0f, std::min(30.0f, temp_c));
        float precip_mm = cells[i].precipitation * 8000.0f;
        float pet = biotemp * 58.93f;
        float pet_ratio = pet / (precip_mm + 1e-3f);

        BiomeType b;
        if (biotemp < 1.5f) b = BiomeType::ICE;
        else if (biotemp < 3.0f) b = BiomeType::TUNDRA;
        else if (biotemp < 12.0f) {
            if (pet_ratio > 2.0f) b = BiomeType::DESERT;
            else if (pet_ratio > 1.0f) b = BiomeType::STEPPE;
            else if (pet_ratio > 0.5f) b = BiomeType::BOREAL_FOREST;
            else b = BiomeType::TEMPERATE_RAINFOREST;
        } else if (biotemp < 24.0f) {
            if (pet_ratio > 2.0f) {
                if (precip_mm < 250.0f) b = BiomeType::DESERT;
                else b = BiomeType::THORN_SCRUB;
            }
            else if (pet_ratio > 1.0f) b = BiomeType::MEDITERRANEAN;
            else if (pet_ratio > 0.5f) b = BiomeType::TEMPERATE_FOREST;
            else b = BiomeType::TEMPERATE_RAINFOREST;
        } else {
            if (pet_ratio > 4.0f) b = BiomeType::DESERT;
            else if (pet_ratio > 2.0f) b = BiomeType::THORN_SCRUB;
            else if (pet_ratio > 1.0f) b = BiomeType::SAVANNA;
            else if (pet_ratio > 0.5f) b = BiomeType::TROPICAL_DRY_FOREST;
            else b = BiomeType::RAINFOREST;
        }
        ideal_biomes[i] = b;
    }

    // Assign ideal biomes directly (Removing Voronoi clustering to avoid mega-deserts)
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation > params.effective_sea_level()) {
            cells[i].biome = ideal_biomes[i];
        }
    }

    // Step 3: Cellular Automata Smoothing (2 iterations)
    for (int iter = 0; iter < 2; ++iter) {
        std::vector<BiomeType> next_biomes(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].elevation <= params.effective_sea_level()) {
                next_biomes[i] = BiomeType::OCEAN;
                continue;
            }
            
            // Count neighbor biomes
            std::vector<int> counts(static_cast<int>(BiomeType::THORN_SCRUB) + 1, 0);
            for (size_t nid : cells[i].neighbors) {
                if (cells[nid].elevation > params.effective_sea_level()) {
                    counts[static_cast<int>(cells[nid].biome)]++;
                }
            }
            
            // Find most common neighbor biome
            int max_count = 0;
            BiomeType most_common = cells[i].biome; // Default to self
            for (int b = 0; b <= static_cast<int>(BiomeType::THORN_SCRUB); ++b) {
                if (counts[b] > max_count) {
                    max_count = counts[b];
                    most_common = static_cast<BiomeType>(b);
                }
            }
            next_biomes[i] = most_common;
        }
        
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].elevation > params.effective_sea_level()) {
                cells[i].biome = next_biomes[i];
            }
        }
    }
}

} // namespace Ravis
