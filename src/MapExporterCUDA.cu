#include "../include/MathUtils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace Ravis {

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA Kernel to compute pixel-to-cell mapping
__global__ void compute_pixel_map_kernel(
    int width, int height, 
    int num_cells, 
    const float* cell_pos_x, const float* cell_pos_y, const float* cell_pos_z, 
    size_t* pixel_map) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double v = static_cast<double>(y) / (height - 1);
    double lat = 1.57079632679489661923 - v * 3.14159265358979323846; // PI/2 - v*PI

    double u = static_cast<double>(x) / (width - 1);
    double lon = u * 6.28318530717958647692 - 3.14159265358979323846; // u*2*PI - PI

    // Spherical to Cartesian
    double cosLat = cos(lat);
    double px = cosLat * cos(lon);
    double py = cosLat * sin(lon);
    double pz = sin(lat);

    double minDist = 1e30;
    size_t closestId = 0;

    // Linear search over all cells
    for (int i = 0; i < num_cells; ++i) {
        double dx = px - cell_pos_x[i];
        double dy = py - cell_pos_y[i];
        double dz = pz - cell_pos_z[i];
        double distSq = dx*dx + dy*dy + dz*dz;
        if (distSq < minDist) {
            minDist = distSq;
            closestId = i;
        }
    }

    pixel_map[y * width + x] = closestId;
}

// Host function to invoke the CUDA kernel
void buildPixelToCellMapCUDA(int width, int height, int num_cells, const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z, std::vector<size_t>& map) {
    float *d_pos_x, *d_pos_y, *d_pos_z;
    size_t *d_map;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_map, width * height * sizeof(size_t)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    compute_pixel_map_kernel<<<numBlocks, blockSize>>>(width, height, num_cells, d_pos_x, d_pos_y, d_pos_z, d_map);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(map.data(), d_map, width * height * sizeof(size_t), cudaMemcpyDeviceToHost));

    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_map);
}

} // namespace Ravis
