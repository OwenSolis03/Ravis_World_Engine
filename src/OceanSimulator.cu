#include "OceanSimulator.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

namespace Ravis {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void ocean_munk_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* elevation,
    const float* wind_x, const float* wind_y, const float* wind_z,
    float* stream_func, float* next_stream_func,
    const int* neighbors, const int* num_neighbors,
    float sea_level
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    if (elevation[i] >= sea_level) {
        next_stream_func[i] = 0.0f; // No currents on land
        return;
    }

    float pz = pos_z[i]; // roughly sine of latitude (Coriolis parameter f ~ 2 * Omega * sin(lat))
    float beta = sqrtf(1.0f - pz * pz); // derivative of f (cos(lat))

    // A simplified stream function solver using Jacobi relaxation
    // The wind stress curl drives the gyres.
    // Westward intensification is modeled by biasing the diffusion based on the beta effect.
    
    float wind_curl = wind_x[i] * pz - wind_y[i] * pz; // Simplified mock wind curl
    
    int nn = num_neighbors[i];
    float sum_psi = 0.0f;
    float sum_weights = 0.0f;

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        if (elevation[nid] < sea_level) {
            float dx = pos_x[nid] - pos_x[i];
            
            // Beta effect: shift the weight to the west (dx < 0)
            float weight = 1.0f - dx * beta * 2.0f; 
            if (weight < 0.1f) weight = 0.1f;

            sum_psi += stream_func[nid] * weight;
            sum_weights += weight;
        }
    }

    if (sum_weights > 0.0f) {
        // Source term is wind_curl
        float new_psi = (sum_psi + wind_curl * 0.1f) / sum_weights;
        next_stream_func[i] = stream_func[i] * 0.9f + new_psi * 0.1f; // Relaxation
    } else {
        next_stream_func[i] = 0.0f;
    }
}

// ============================================================================
// Host Methods
// ============================================================================

OceanSimulator::OceanSimulator(GoldbergPolyhedron& p) : planet(p) {}

void OceanSimulator::simulateCurrents(const SimulationParameters& params) {
    std::cout << "Simulating Ocean Currents (Munk Model)..." << std::endl;
    
    // In a full implementation, we'd copy elevation, wind, and run the Munk solver loop
    // Since we don't have wind data fully baked into the planet cells here yet (it's in Atmosphere),
    // we can either move wind to PlanetData or approximate it.
    
    // For now, this is a stub that will be fleshed out when Atmosphere Simulator is fully coupled.
    std::cout << "Ocean currents step complete." << std::endl;
}

} // namespace Ravis
