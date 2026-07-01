#include "../include/ErosionSimulator.h"
#include "../include/MathUtils.h"
#include <random>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Ravis {

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void setup_curand_kernel(curandState *state, unsigned long seed, int num_drops) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < num_drops) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void erosion_drop_kernel(
    int num_drops, int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation, const float* moisture, RockType* bedrock,
    const int* neighbors, const int* num_neighbors,
    float sea_level, float erosion_rate,
    curandState *state) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_drops) return;

    curandState localState = state[id];
    
    const float min_slope = 100.0f;
    const float capacity_factor = 0.005f;
    const float deposition_rate = 0.1f;
    const float evaporation_rate = 0.02f;
    const float friction = 0.05f;
    const int max_lifetime = 30;

    int currentId = curand(&localState) % num_cells;
    int retries = 0;
    while (elevation[currentId] <= sea_level && retries < 100) {
        currentId = curand(&localState) % num_cells;
        retries++;
    }
    if (retries >= 100) return;

    float water = moisture[currentId] * 10.0f;
    float velocity = 1.0f;
    float sediment = 0.0f;

    for (int lifetime = 0; lifetime < max_lifetime; ++lifetime) {
        int lowestNeighbor = currentId;
        float minElev = elevation[currentId];
        
        int nn = num_neighbors[currentId];
        for (int i = 0; i < nn; ++i) {
            int nid = neighbors[currentId * 6 + i];
            float nElev = elevation[nid];
            if (nElev < minElev) {
                minElev = nElev;
                lowestNeighbor = nid;
            }
        }

        if (lowestNeighbor == currentId) {
            atomicAdd(&elevation[currentId], sediment);
            break;
        }
        
        float elevDiff = elevation[currentId] - elevation[lowestNeighbor];
        float dx = pos_x[currentId] - pos_x[lowestNeighbor];
        float dy = pos_y[currentId] - pos_y[lowestNeighbor];
        float dz = pos_z[currentId] - pos_z[lowestNeighbor];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        float slope = elevDiff / (dist + 1e-6f);

        float capacity = fmaxf(slope, min_slope) * velocity * water * capacity_factor;

        float base_erosion = 0.1f;
        RockType rock = bedrock[currentId];
        if (rock == RockType::BASALT) base_erosion = 0.05f;
        else if (rock == RockType::GRANITE) base_erosion = 0.02f;
        else if (rock == RockType::SANDSTONE) base_erosion = 0.20f;
        else if (rock == RockType::SHALE_LIMESTONE) base_erosion = 0.30f;
        else if (rock == RockType::METAMORPHIC) base_erosion = 0.03f;
        
        float current_erosion_rate = base_erosion * erosion_rate;

        if (sediment > capacity) {
            float amount = (sediment - capacity) * deposition_rate;
            sediment -= amount;
            atomicAdd(&elevation[currentId], amount);
            
            if (amount > 10.0f) {
                if (elevation[currentId] <= sea_level) bedrock[currentId] = RockType::SHALE_LIMESTONE;
                else bedrock[currentId] = RockType::SANDSTONE;
            }
        } else {
            float amount = fminf((capacity - sediment) * current_erosion_rate, elevDiff);
            sediment += amount;
            atomicAdd(&elevation[currentId], -amount);
        }

        currentId = lowestNeighbor;
        velocity = sqrtf(velocity * velocity + elevDiff * 9.8f) * (1.0f - friction);
        water *= (1.0f - evaporation_rate);

        if (elevation[currentId] <= sea_level) {
            atomicAdd(&elevation[currentId], sediment);
            bedrock[currentId] = RockType::SHALE_LIMESTONE;
            break;
        }
        if (water < 0.01f) {
            atomicAdd(&elevation[currentId], sediment);
            bedrock[currentId] = RockType::SANDSTONE;
            break;
        }
    }
    state[id] = localState;
}

__global__ void coastal_erosion_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation, RockType* bedrock,
    const float* wind_u, const float* wind_v, const float* wind_w,
    const int* neighbors, const int* num_neighbors,
    float sea_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    if (elevation[i] <= sea_level) return; // Only process land cells

    bool is_coast = false;
    int ocean_neighbor = -1;
    int nn = num_neighbors[i];
    for(int n=0; n<nn; ++n) {
        int nid = neighbors[i * 6 + n];
        if (elevation[nid] <= sea_level) {
            is_coast = true;
            ocean_neighbor = nid;
            break;
        }
    }

    if (is_coast) {
        // Wind dot product towards land
        float dx = pos_x[i] - pos_x[ocean_neighbor];
        float dy = pos_y[i] - pos_y[ocean_neighbor];
        float dz = pos_z[i] - pos_z[ocean_neighbor];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        dx /= dist; dy /= dist; dz /= dist;

        float dot = wind_u[ocean_neighbor]*dx + wind_v[ocean_neighbor]*dy + wind_w[ocean_neighbor]*dz;
        if (dot > 0.0f) { // Wind blowing towards land
            float erosion_amount = dot * 5.0f; // Erode up to 5m per tick based on wave energy
            if (elevation[i] - erosion_amount > sea_level) {
                atomicAdd(&elevation[i], -erosion_amount);
                atomicAdd(&elevation[ocean_neighbor], erosion_amount * 0.5f); // Create continental shelf
            }
        }
    }
}

ErosionSimulator::ErosionSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void ErosionSimulator::simulateErosion(int numDrops, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    int num_cells = cells.size();

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_elevation(num_cells), h_moisture(num_cells);
    std::vector<float> h_wind_u(num_cells), h_wind_v(num_cells), h_wind_w(num_cells);
    std::vector<RockType> h_bedrock(num_cells);
    std::vector<int> h_neighbors(num_cells * 6, -1);
    std::vector<int> h_num_neighbors(num_cells, 0);

    for (int i = 0; i < num_cells; ++i) {
        h_pos_x[i] = cells[i].position.x;
        h_pos_y[i] = cells[i].position.y;
        h_pos_z[i] = cells[i].position.z;
        h_elevation[i] = cells[i].elevation;
        h_moisture[i] = cells[i].moisture;
        h_bedrock[i] = cells[i].bedrock;
        h_wind_u[i] = cells[i].wind_velocity.x;
        h_wind_v[i] = cells[i].wind_velocity.y;
        h_wind_w[i] = cells[i].wind_velocity.z;

        h_num_neighbors[i] = cells[i].neighbors.size();
        for (size_t n = 0; n < cells[i].neighbors.size(); ++n) {
            h_neighbors[i * 6 + n] = cells[i].neighbors[n];
        }
    }

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_elevation, *d_moisture, *d_wind_u, *d_wind_v, *d_wind_w;
    RockType *d_bedrock;
    int *d_neighbors, *d_num_neighbors;
    curandState *d_state;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elevation, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_moisture, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bedrock, num_cells * sizeof(RockType)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_state, numDrops * sizeof(curandState)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_elevation, h_elevation.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_moisture, h_moisture.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_u, h_wind_u.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_v, h_wind_v.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_w, h_wind_w.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bedrock, h_bedrock.data(), num_cells * sizeof(RockType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocksDrops = (numDrops + blockSize - 1) / blockSize;
    int numBlocksCells = (num_cells + blockSize - 1) / blockSize;

    setup_curand_kernel<<<numBlocksDrops, blockSize>>>(d_state, params.seed, numDrops);
    CHECK_CUDA(cudaDeviceSynchronize());

    erosion_drop_kernel<<<numBlocksDrops, blockSize>>>(
        numDrops, num_cells,
        d_pos_x, d_pos_y, d_pos_z,
        d_elevation, d_moisture, d_bedrock,
        d_neighbors, d_num_neighbors,
        params.sea_level, params.erosion_rate,
        d_state
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    coastal_erosion_kernel<<<numBlocksCells, blockSize>>>(
        num_cells, d_pos_x, d_pos_y, d_pos_z,
        d_elevation, d_bedrock, d_wind_u, d_wind_v, d_wind_w,
        d_neighbors, d_num_neighbors, params.sea_level
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_elevation.data(), d_elevation, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_bedrock.data(), d_bedrock, num_cells * sizeof(RockType), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cells; ++i) {
        cells[i].elevation = h_elevation[i];
        cells[i].bedrock = h_bedrock[i];
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_elevation); cudaFree(d_moisture);
    cudaFree(d_wind_u); cudaFree(d_wind_v); cudaFree(d_wind_w);
    cudaFree(d_bedrock);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
    cudaFree(d_state);
}

} // namespace Ravis
