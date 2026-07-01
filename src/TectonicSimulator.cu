#include "../include/TectonicSimulator.h"
#include "../include/MathUtils.h"
#include <random>
#include <queue>
#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>

namespace Ravis {

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Hash-based 3D noise — NON-SEPARABLE, no mirror symmetry
// Uses dot products with irrational vectors to mix all coordinates before sin()
// ============================================================================
__device__ float hash_noise3d(float x, float y, float z) {
    // Mix all three coordinates via dot products with large primes
    float d1 = x * 127.1f + y * 311.7f + z * 74.7f;
    float d2 = x * 269.5f + y * 183.3f + z * 246.1f;
    float d3 = x * 419.2f + y * 371.9f + z * 128.9f;
    
    float h1 = sinf(d1) * 43758.5453f;
    float h2 = sinf(d2) * 22578.1459f;
    float h3 = sinf(d3) * 10003.2987f;
    
    // Fractional parts give pseudo-random [-1, 1]
    h1 = h1 - floorf(h1);
    h2 = h2 - floorf(h2);
    h3 = h3 - floorf(h3);
    
    return sinf(h1 * 6.2832f + h2 * 3.1416f + h3 * 1.5708f);
}

// FBM (Fractal Brownian Motion) — multi-octave noise for natural terrain
__device__ float hash_fbm3d(float x, float y, float z, int octaves) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float total_amp = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        value += hash_noise3d(x * frequency, y * frequency, z * frequency) * amplitude;
        total_amp += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value / total_amp;
}

// Ridge Noise (Fractal blending with 1.0 - abs(noise))
__device__ float hash_ridge3d(float x, float y, float z, int octaves) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float total_amp = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        float n = hash_noise3d(x * frequency, y * frequency, z * frequency);
        // Ridge formula
        n = 1.0f - fabsf(n);
        n *= n; // Sharpen ridges
        value += n * amplitude;
        total_amp += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value / total_amp;
}

// CPU-side equivalents for host code
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

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void tectonic_bisector_stress_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* stress, const int* plate_id,
    int numPlates, 
    const float* center_x, const float* center_y, const float* center_z,
    const float* axis_x, const float* axis_y, const float* axis_z,
    const float* speeds)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    
    float c1x = center_x[pid], c1y = center_y[pid], c1z = center_z[pid];
    float dx1 = px - c1x, dy1 = py - c1y, dz1 = pz - c1z;
    float dist1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1 + 1e-6f);
    
    float min_bisector_dist = 1e10f;
    int closest_pid = -1;
    
    for (int p = 0; p < numPlates; ++p) {
        if (p == pid) continue;
        float c2x = center_x[p], c2y = center_y[p], c2z = center_z[p];
        float dx2 = px - c2x, dy2 = py - c2y, dz2 = pz - c2z;
        float dist2 = sqrtf(dx2*dx2 + dy2*dy2 + dz2*dz2 + 1e-6f);
        
        float b_dist = (dist2 - dist1) * 0.5f;
        if (b_dist < min_bisector_dist && b_dist >= 0.0f) {
            min_bisector_dist = b_dist;
            closest_pid = p;
        }
    }
    
    if (closest_pid != -1) {
        // Calculate velocity of my plate at this position
        float v1x = axis_y[pid] * pz - axis_z[pid] * py;
        float v1y = axis_z[pid] * px - axis_x[pid] * pz;
        float v1z = axis_x[pid] * py - axis_y[pid] * px;
        v1x *= speeds[pid]; v1y *= speeds[pid]; v1z *= speeds[pid];
        
        // Calculate velocity of other plate at this position
        float v2x = axis_y[closest_pid] * pz - axis_z[closest_pid] * py;
        float v2y = axis_z[closest_pid] * px - axis_x[closest_pid] * pz;
        float v2z = axis_x[closest_pid] * py - axis_y[closest_pid] * px;
        v2x *= speeds[closest_pid]; v2y *= speeds[closest_pid]; v2z *= speeds[closest_pid];
        
        float rel_vx = v2x - v1x;
        float rel_vy = v2y - v1y;
        float rel_vz = v2z - v1z;
        
        // Direction from closest seed to my seed (approx boundary normal)
        float nx = c1x - center_x[closest_pid];
        float ny = c1y - center_y[closest_pid];
        float nz = c1z - center_z[closest_pid];
        float nlen = sqrtf(nx*nx + ny*ny + nz*nz + 1e-6f);
        nx /= nlen; ny /= nlen; nz /= nlen;
        
        // Dot product to see if colliding (-) or diverging (+)
        float dot = nx * rel_vx + ny * rel_vy + nz * rel_vz;
        
        // Convert distance to a stress falloff
        // If we are exactly at the bisector, min_bisector_dist = 0
        // We want stress to be max at 0 and decay as distance increases
        // Normal sphere radius is 1.0. A plate might be 0.5 across.
        float falloff = max(0.0f, 1.0f - (min_bisector_dist * 8.0f));
        stress[i] = -dot * 100.0f * falloff;
    } else {
        stress[i] = 0.0f;
    }
}

__global__ void tectonic_calc_stress_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* stress, const int* plate_id,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const int* neighbors, const int* num_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float vx = vel_x[i], vy = vel_y[i], vz = vel_z[i];

    float max_stress = 0.0f;
    int nn = num_neighbors[i];

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        if (pid != plate_id[nid]) {
            float dx = pos_x[nid] - px;
            float dy = pos_y[nid] - py;
            float dz = pos_z[nid] - pz;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
            dx /= dist; dy /= dist; dz /= dist;

            float rel_vx = vel_x[nid] - vx;
            float rel_vy = vel_y[nid] - vy;
            float rel_vz = vel_z[nid] - vz;

            float dot = dx * rel_vx + dy * rel_vy + dz * rel_vz;
            
            // Negative dot = convergent (moving towards each other)
            // Positive dot = divergent (moving away)
            if (fabsf(dot) > fabsf(max_stress)) {
                max_stress = -dot; // Positive stress for convergence, negative for divergence
            }
        }
    }
    
    // Scale up the stress value so it diffuses nicely
    stress[i] = max_stress * 100.0f;
}

__global__ void tectonic_diffuse_stress_kernel(
    int num_cells,
    const float* stress, float* next_stress,
    const int* neighbors, const int* num_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int nn = num_neighbors[i];
    float sum = stress[i];
    float count = 1.0f;

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        sum += stress[nid];
        count += 1.0f;
    }
    
    // Smooth diffusion
    next_stress[i] = (sum / count);
}

// "Motor del Tiempo" - Tectonic Drift Boundary Assimilation
__global__ void tectonic_drift_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const int* plate_id, int* next_plate_id,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const uint8_t* is_oceanic, uint8_t* next_is_oceanic,
    const float* elev, float* next_elev,
    const int* neighbors, const int* num_neighbors,
    int seed_offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    int nn = num_neighbors[i];
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];

    // Default to keeping current state
    next_plate_id[i] = pid;
    next_is_oceanic[i] = is_oceanic[i];
    next_elev[i] = elev[i];

    bool is_boundary = false;
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        if (plate_id[nid] != pid) {
            is_boundary = true;
            break;
        }
    }
    
    if (!is_boundary) return;

    // Evaluate neighbors to see if any are moving into our cell
    float max_attack = 0.0f;
    int attacker_pid = -1;
    bool attacker_oceanic = false;
    float attacker_elev = 0.0f;

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        int npid = plate_id[nid];
        if (npid != pid) {
            // Velocity of neighbor
            float nvx = vel_x[nid], nvy = vel_y[nid], nvz = vel_z[nid];
            
            // Direction FROM neighbor TO me
            float dx = px - pos_x[nid];
            float dy = py - pos_y[nid];
            float dz = pz - pos_z[nid];
            float dist = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
            dx /= dist; dy /= dist; dz /= dist;
            
            // Attack strength is dot product (positive means moving towards me)
            float attack = nvx * dx + nvy * dy + nvz * dz;
            if (attack > max_attack) {
                max_attack = attack;
                attacker_pid = npid;
                attacker_oceanic = is_oceanic[nid];
                attacker_elev = elev[nid];
            }
        }
    }
    
    if (max_attack > 0.0f) {
        // Pseudo-random hash for this cell
        float r = hash_noise3d(px + seed_offset, py - seed_offset, pz * seed_offset); // [-1, 1]
        
        // Scale probability
        float prob = max_attack * 1500.0f; // Scale up velocity to probability
        if ((r + 1.0f) * 0.5f < prob) { 
            next_plate_id[i] = attacker_pid;
            next_is_oceanic[i] = attacker_oceanic;
            next_elev[i] = attacker_elev; // Inherit base elevation so we don't leave spikes
        }
    }
}

__global__ void tectonic_apply_stress_kernel(
    int num_cells,
    float* elevation, const float* stress, const uint8_t* is_oceanic,
    float orogenesis_factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float s = stress[i];
    bool oceanic = is_oceanic[i];
    
    if (fabsf(s) > 0.001f) {
        // Smoothstep falloff mapping
        float mag = fabsf(s);
        float falloff = mag * mag * (3.0f - 2.0f * mag); // Smooth curve
        
        float delta = 0.0f;
        if (s > 0.0f) { // Convergence (Mountains / Subduction)
            if (oceanic) delta = -falloff * 8000.0f * orogenesis_factor; // Oceanic Trench (Subduction)
            else delta = falloff * 20000.0f * orogenesis_factor; // Continental Mountains (Himalayas/Andes)
        } else { // Divergence (Trenches / Ridges)
            if (oceanic) delta = -falloff * 4000.0f; // Mid-Ocean Ridges
            else delta = -falloff * 2000.0f; // Rift valleys
        }
        
        elevation[i] += delta;
    }
}

// Smoothing kernel
__global__ void tectonic_smooth_kernel(
    int num_cells,
    float* elevation, float* next_elevation,
    const int* plate_id,
    const int* neighbors, const int* num_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id ? plate_id[i] : -1;
    bool is_boundary = false;
    int nn = num_neighbors[i];

    float sum = elevation[i];
    float count = 1.0f;

    if (plate_id) {
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[i * 6 + n];
            if (plate_id[nid] != pid) {
                is_boundary = true;
                break;
            }
        }
    }

    if (!is_boundary) {
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[i * 6 + n];
            sum += elevation[nid];
            count += 1.0f;
        }
        next_elevation[i] = elevation[i] * 0.8f + (sum / count) * 0.2f;
    } else {
        next_elevation[i] = elevation[i];
    }
}

// Backward Advection Kernel to move plates without tearing (uses sub-stepping to fix smearing)
__global__ void tectonic_apply_advection_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* elev_in, float* elev_out,
    const int* plate_id_in, int* plate_id_out,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const uint8_t* is_oceanic_in, uint8_t* is_oceanic_out,
    const float* age_in, float* age_out,
    const float* thick_in, float* thick_out,
    const float* vx_in, float* vx_out,
    const float* vy_in, float* vy_out,
    const float* vz_in, float* vz_out,
    const int* neighbors, const int* num_neighbors,
    float dt_age, float subsidence_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float vx = vel_x[i], vy = vel_y[i], vz = vel_z[i];

    // Hill-climbing to find the backward position with temporal sub-stepping
    int best_j = i;
    for (int step = 1; step <= 5; ++step) {
        float t = step / 5.0f;
        float sub_fx = px - vx * t; // BACKWARD step
        float sub_fy = py - vy * t;
        float sub_fz = pz - vz * t;
        float sub_inv = rsqrtf(sub_fx*sub_fx + sub_fy*sub_fy + sub_fz*sub_fz);
        sub_fx *= sub_inv; sub_fy *= sub_inv; sub_fz *= sub_inv;

        float best_dot = pos_x[best_j] * sub_fx + pos_y[best_j] * sub_fy + pos_z[best_j] * sub_fz;
        bool improved = true;
        int max_steps = 10;
        
        while (improved && max_steps > 0) {
            improved = false;
            max_steps--;
            int nn = num_neighbors[best_j];
            for (int n = 0; n < nn; ++n) {
                int nid = neighbors[best_j * 6 + n];
                float d = pos_x[nid] * sub_fx + pos_y[nid] * sub_fy + pos_z[nid] * sub_fz;
                if (d > best_dot) {
                    best_dot = d;
                    best_j = nid;
                    improved = true;
                }
            }
        }
    }

    // Check if it's a true divergence gap.
    // If we looked backwards and found a cell, but that cell is moving AWAY from us, it's a gap!
    float fwd_x = pos_x[best_j] + vel_x[best_j];
    float fwd_y = pos_y[best_j] + vel_y[best_j];
    float fwd_z = pos_z[best_j] + vel_z[best_j];
    float inv_fwd = rsqrtf(fwd_x*fwd_x + fwd_y*fwd_y + fwd_z*fwd_z);
    fwd_x *= inv_fwd; fwd_y *= inv_fwd; fwd_z *= inv_fwd;
    
    // Check if the source cell's forward destination lands near `i`
    float dot_fwd = fwd_x * px + fwd_y * py + fwd_z * pz;
    bool is_gap = (dot_fwd < 0.999f) && (plate_id_in[best_j] != plate_id_in[i]);

    if (!is_gap) {
        elev_out[i] = elev_in[best_j];
        plate_id_out[i] = plate_id_in[best_j];
        is_oceanic_out[i] = is_oceanic_in[best_j];
        age_out[i] = age_in[best_j];
        thick_out[i] = thick_in[best_j];
        vx_out[i] = vx_in[best_j];
        vy_out[i] = vy_in[best_j];
        vz_out[i] = vz_in[best_j];
        
        if (is_oceanic_out[i]) {
            age_out[i] += dt_age;
            float target_depth = 2500.0f + subsidence_rate * sqrtf(max(0.0f, age_out[i]));
            elev_out[i] = -target_depth;
        }
    } else {
        // Divergence gap: spawn oceanic crust
        elev_out[i] = -2500.0f;
        is_oceanic_out[i] = 1;
        age_out[i] = 0.0f;
        thick_out[i] = 7.0f;
        plate_id_out[i] = plate_id_in[i];
        vx_out[i] = vx_in[i];
        vy_out[i] = vy_in[i];
        vz_out[i] = vz_in[i];
    }
}

// Hotspot kernel — uses hash noise for organic island shapes
__global__ void tectonic_hotspot_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation, const uint8_t* is_oceanic,
    int num_hotspots, const float* hs_x, const float* hs_y, const float* hs_z, 
    const float* hs_str, const float* hs_radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float local_elev = elevation[i];

    for (int h = 0; h < num_hotspots; ++h) {
        float dx = px - hs_x[h];
        float dy = py - hs_y[h];
        float dz = pz - hs_z[h];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);

        float radius = hs_radius[h];
        float check_radius = radius * 1.5f;

        if (dist < check_radius) {
            float noise = hash_fbm3d(px * 15.0f, py * 15.0f, pz * 15.0f, 3);
            float dist_perturbed = dist * (1.0f - noise * 0.5f);
            
            if (dist_perturbed < radius) {
                float intensity = 1.0f - dist_perturbed / radius;
                intensity *= intensity; // Quadratic falloff for more realistic shape
                local_elev += hs_str[h] * intensity * (1.0f + fabsf(noise) * 0.4f);
            }
        }
    }
    
    elevation[i] = local_elev;
}

// High-frequency noise overlay kernel for continuous terrain detail (with per-plate offset)
__global__ void tectonic_fbm_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation, const float* stress, const uint8_t* is_oceanic, const int* plate_id
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float s = fabsf(stress[i]); // We use tectonic stress to scale mountains
    
    // Global procedural noise without plate ID offset to ensure completely continuous height gradients
    float spx = px * 2.0f;
    float spy = py * 2.0f;
    float spz = pz * 2.0f;
    
    // Low frequency continents shaping (Octave 1 suppression)
    float low_noise = hash_fbm3d(spx * 2.0f, spy * 2.0f, spz * 2.0f, 4);
    // Suppress central mountains: limit the maximum height from the base noise
    float plateau_limit = 0.3f;
    if (low_noise > plateau_limit) {
        low_noise = plateau_limit + (low_noise - plateau_limit) * 0.1f; // Clamp / Plateau
    }
    
    // High frequency detail (Ridge noise multiplied by stress)
    float ridge_detail = hash_ridge3d(spx * 30.0f, spy * 30.0f, spz * 30.0f, 6);
    float ridge_hills = hash_ridge3d(spx * 8.0f, spy * 8.0f, spz * 8.0f, 5);
    
    // Scale ridges by tectonic stress (s usually maxes around 1.0 - 5.0)
    float stress_multiplier = min(1.0f, s * 0.2f);
    
    float current_elev = elevation[i];
    
    if (!is_oceanic[i]) {
        // Continental terrain
        current_elev += low_noise * 800.0f; // Base continent shape (plateaus)
        
        // Add mountains ONLY where stress is high
        current_elev += ridge_hills * 1500.0f * stress_multiplier;
        current_elev += ridge_detail * 800.0f * stress_multiplier;
        
        // Add a tiny bit of basic noise everywhere so plains aren't perfectly smooth
        current_elev += hash_fbm3d(spx * 20.0f, spy * 20.0f, spz * 20.0f, 3) * 50.0f;
        
        // Ensure no terrain goes below sea level if it's continental
        if (current_elev < 50.0f) current_elev = 50.0f;
    } else {
        // Oceanic terrain
        current_elev += low_noise * 300.0f;
        current_elev += ridge_hills * 500.0f * stress_multiplier;
        current_elev += hash_fbm3d(spx * 20.0f, spy * 20.0f, spz * 20.0f, 3) * 100.0f;
        
        // Ensure trenches stay deep
        if (current_elev > -100.0f) current_elev = -100.0f;
    }
    
    elevation[i] = current_elev;
}

// ============================================================================
// Host methods
// ============================================================================

TectonicSimulator::TectonicSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void TectonicSimulator::generatePlates(int numPlates, const SimulationParameters& params) {
    // TODO: Add support for subdivision_level = 13 (1km hexagons, ~510M cells).
    // This requires >30GB VRAM and supercomputer infrastructure (MPI + Multi-GPU)
    // as it vastly exceeds a single consumer GPU's memory limits for tectonic tracking.
    auto& cells = planet.getCells();
    if (cells.empty()) return;

    std::mt19937 rng(params.seed);
    std::uniform_int_distribution<size_t> distNode(0, cells.size() - 1);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);

    plates.resize(numPlates);

    std::queue<size_t> bfsQueue;
    std::vector<bool> visited(cells.size(), false);

    // --- Primary clustering: generate a "Pangea pole" for continental seed attraction ---
    Vector3 pangea_pole(distReal(rng), distReal(rng), distReal(rng));
    pangea_pole = Math::normalize(pangea_pole);
    
    // Secondary cluster pole (opposite hemisphere for America-style fragments)
    Vector3 secondary_pole(-pangea_pole.x + distReal(rng) * 0.5,
                           -pangea_pole.y + distReal(rng) * 0.5,
                           -pangea_pole.z + distReal(rng) * 0.5);
    secondary_pole = Math::normalize(secondary_pole);

    float speed_scale = params.tectonic_speed();

    int numMajor = std::max(1, static_cast<int>(numPlates * 0.7f));
    int numMinor = numPlates - numMajor;

    for (int i = 0; i < numMajor; ++i) {
        // Decide continental vs oceanic
        bool is_oceanic_plate = distProb(rng) > params.crust_fraction;
        
        size_t seedId;
        if (!is_oceanic_plate) {
            // Continental plates: bias seed placement toward clustering poles
            float cluster_strength = params.primary_clustering;
            float secondary_strength = params.secondary_clustering;
            
            size_t best = distNode(rng);
            float best_score = 1e10f;
            int attempts = static_cast<int>(50 * cluster_strength) + 1;
            
            for (int a = 0; a < attempts; ++a) {
                size_t candidate = distNode(rng);
                if (visited[candidate]) continue;
                
                float dx = cells[candidate].position.x - pangea_pole.x;
                float dy = cells[candidate].position.y - pangea_pole.y;
                float dz = cells[candidate].position.z - pangea_pole.z;
                float dist_primary = std::sqrt(dx*dx + dy*dy + dz*dz);
                
                float dx2 = cells[candidate].position.x - secondary_pole.x;
                float dy2 = cells[candidate].position.y - secondary_pole.y;
                float dz2 = cells[candidate].position.z - secondary_pole.z;
                float dist_secondary = std::sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                
                float score = dist_primary * cluster_strength + 
                              dist_secondary * secondary_strength * 0.5f +
                              distProb(rng) * (1.0f - cluster_strength);
                
                if (score < best_score) {
                    best_score = score;
                    best = candidate;
                }
            }
            seedId = best;
        } else {
            do { seedId = distNode(rng); } while (visited[seedId]);
        }

        while (visited[seedId]) { seedId = distNode(rng); }
        
        visited[seedId] = true;
        cells[seedId].plate_id = i;
        bfsQueue.push(seedId);

        plates[i].is_oceanic = is_oceanic_plate;
        plates[i].center = cells[seedId].position;
        Vector3 axis(distReal(rng), distReal(rng), distReal(rng));
        plates[i].rotation_axis = Math::normalize(axis);
        
        if (plates[i].is_oceanic) {
            plates[i].angular_speed = distProb(rng) * speed_scale * 0.2f + 0.005f;
        } else {
            plates[i].angular_speed = distProb(rng) * speed_scale * 0.1f + 0.001f;
        }
    }

    // Grow Major Plates
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

    // Find boundaries of major plates
    std::vector<size_t> boundary_cells;
    for (size_t i = 0; i < cells.size(); ++i) {
        int pid = cells[i].plate_id;
        bool is_boundary = false;
        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].plate_id != pid) {
                is_boundary = true;
                break;
            }
        }
        if (is_boundary) boundary_cells.push_back(i);
    }

    // Spawn Minor Plates on boundaries
    if (!boundary_cells.empty()) {
        std::uniform_int_distribution<size_t> distBoundary(0, boundary_cells.size() - 1);
        std::vector<bool> minor_visited(cells.size(), false);
        
        for (int i = numMajor; i < numPlates; ++i) {
            size_t seedId = boundary_cells[distBoundary(rng)];
            
            cells[seedId].plate_id = i;
            minor_visited[seedId] = true;
            bfsQueue.push(seedId);

            plates[i].is_oceanic = distProb(rng) > params.crust_fraction;
            plates[i].center = cells[seedId].position;
            Vector3 axis(distReal(rng), distReal(rng), distReal(rng));
            plates[i].rotation_axis = Math::normalize(axis);
            
            // Minor plates tend to move faster (fragments)
            if (plates[i].is_oceanic) {
                plates[i].angular_speed = distProb(rng) * speed_scale * 0.3f + 0.01f;
            } else {
                plates[i].angular_speed = distProb(rng) * speed_scale * 0.15f + 0.005f;
            }
        }

        // Grow Minor Plates for a limited number of steps
        int max_minor_steps = static_cast<int>(std::sqrt(cells.size()) * 0.05f); // Limit growth
        int current_step = 0;
        
        while (!bfsQueue.empty() && current_step < max_minor_steps) {
            int level_size = bfsQueue.size();
            for (int k = 0; k < level_size; ++k) {
                size_t currentId = bfsQueue.front();
                bfsQueue.pop();

                int currentPlate = cells[currentId].plate_id;

                for (size_t neighborId : cells[currentId].neighbors) {
                    if (!minor_visited[neighborId]) {
                        minor_visited[neighborId] = true;
                        cells[neighborId].plate_id = currentPlate;
                        bfsQueue.push(neighborId);
                    }
                }
            }
            current_step++;
        }
        
        // Clear remaining queue
        while (!bfsQueue.empty()) bfsQueue.pop();
    }

    std::uniform_real_distribution<float> distAge(10.0f, 200.0f); 
    std::uniform_real_distribution<float> distOceanVar(0.0f, 500.0f);
    std::uniform_real_distribution<float> distContVar(0.0f, 200.0f);
    
    float warp = params.crust_warping;
    
    for (auto& cell : cells) {
        int pid = cell.plate_id;
        cell.is_oceanic = plates[pid].is_oceanic;
        
        if (cell.is_oceanic) {
            cell.elevation = -4000.0f + distOceanVar(rng); 
            cell.bedrock = RockType::BASALT;
            cell.crustal_thickness = 7.0f;  
            cell.crustal_age = distAge(rng); 
        } else {
            // Hash-based Fractal Ridge Blending — NO symmetry
            float px = static_cast<float>(cell.position.x);
            float py = static_cast<float>(cell.position.y);
            float pz = static_cast<float>(cell.position.z);
            
            float ridge_noise = cpu_hash_fbm3d(px * 8.0f, py * 8.0f, pz * 8.0f, 4);
            float ridge = std::abs(ridge_noise);
            
            // Crust warping: additional noise distortion to fragment shapes
            float warp_noise = cpu_hash_noise3d(px * 3.0f + 100.0f, py * 3.0f + 200.0f, pz * 3.0f + 300.0f);
            ridge += std::abs(warp_noise) * warp;
            // Flatten plains by applying a power curve. Low values stay low, high values spike up.
            ridge = std::pow(ridge, 2.5f);
            
            cell.elevation = 100.0f + ridge * 3500.0f + distContVar(rng); 
            cell.bedrock = RockType::GRANITE;
            cell.crustal_thickness = 35.0f + (cell.elevation - 100.0f) / 100.0f; // Isostatic equilibrium
            cell.crustal_age = 2500.0f;     
        }

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
    
    // No longer using blocky BFS uplifts. Replaced by FBM kernel in simulate().
}

void TectonicSimulator::applyTerrainFeatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    std::mt19937 rng(params.seed + 7777);
    std::uniform_int_distribution<size_t> distNode(0, cells.size() - 1);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);
    
    // Helper: BFS radial spread from a seed cell on continental crust
    auto bfs_uplift = [&](size_t seed, float peak_elev, float min_elev, int spread_cells) {
        if (cells[seed].is_oceanic) return;
        
        std::queue<size_t> q;
        std::vector<bool> vis(cells.size(), false);
        q.push(seed);
        vis[seed] = true;
        int count = 0;
        
        while (!q.empty() && count < spread_cells) {
            size_t cur = q.front(); q.pop();
            if (cells[cur].is_oceanic) continue;
            
            float t = static_cast<float>(count) / static_cast<float>(spread_cells);
            float uplift = peak_elev * (1.0f - t) + min_elev * t;
            
            // Hash noise for organic shape
            float px = static_cast<float>(cells[cur].position.x);
            float py = static_cast<float>(cells[cur].position.y);
            float pz = static_cast<float>(cells[cur].position.z);
            float noise = cpu_hash_noise3d(px * 10.0f, py * 10.0f, pz * 10.0f);
            uplift *= (0.7f + 0.3f * std::abs(noise));
            
            cells[cur].elevation += uplift;
            cells[cur].crustal_thickness += uplift / 100.0f;
            count++;
            
            for (size_t nid : cells[cur].neighbors) {
                if (!vis[nid] && !cells[nid].is_oceanic) {
                    vis[nid] = true;
                    q.push(nid);
                }
            }
        }
    };
    
    // Old mountains (Appalachian-style): moderate height, large spread
    for (int i = 0; i < params.old_mountain_freq; ++i) {
        size_t seed;
        int tries = 0;
        do { seed = distNode(rng); tries++; } while (cells[seed].is_oceanic && tries < 100);
        bfs_uplift(seed, 1200.0f, 400.0f, 200);
    }
    
    // Old hills (Caledonian-style): low height, medium spread
    for (int i = 0; i < params.old_hill_freq; ++i) {
        size_t seed;
        int tries = 0;
        do { seed = distNode(rng); tries++; } while (cells[seed].is_oceanic && tries < 100);
        bfs_uplift(seed, 600.0f, 150.0f, 100);
    }
    
    // Small uplifts: tiny stochastic bumps
    for (int i = 0; i < params.small_uplift_freq; ++i) {
        size_t seed;
        int tries = 0;
        do { seed = distNode(rng); tries++; } while (cells[seed].is_oceanic && tries < 100);
        bfs_uplift(seed, 400.0f, 100.0f, 30);
    }
    
    // Uplands (Volga-style): very broad, very gentle
    for (int i = 0; i < params.upland_freq; ++i) {
        size_t seed;
        int tries = 0;
        do { seed = distNode(rng); tries++; } while (cells[seed].is_oceanic && tries < 100);
        bfs_uplift(seed, 350.0f, 100.0f, 400);
    }
    
    // Superswells (African-style): massive broad doming, also affects oceanic crust
    for (int i = 0; i < params.superswell_freq; ++i) {
        size_t seed = distNode(rng);
        std::queue<size_t> q;
        std::vector<bool> vis(cells.size(), false);
        q.push(seed);
        vis[seed] = true;
        int count = 0;
        int spread = 600;
        
        while (!q.empty() && count < spread) {
            size_t cur = q.front(); q.pop();
            float t = static_cast<float>(count) / static_cast<float>(spread);
            float uplift = 800.0f * (1.0f - t * t); // Gaussian-like falloff
            cells[cur].elevation += uplift;
            if (!cells[cur].is_oceanic) cells[cur].crustal_thickness += uplift / 200.0f;
            count++;
            
            for (size_t nid : cells[cur].neighbors) {
                if (!vis[nid]) {
                    vis[nid] = true;
                    q.push(nid);
                }
            }
        }
    }
}

void TectonicSimulator::generateHotspots(const SimulationParameters& params) {
    std::mt19937 rng(params.seed + 4242); 
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);

    hotspots.clear();
    
    // Deep plumes (Iceland-style): large radius, very strong, can affect continental crust
    for (int i = 0; i < params.deep_plume_freq; ++i) {
        Vector3 pos(distReal(rng), distReal(rng), distReal(rng));
        Hotspot h;
        h.position = Math::normalize(pos);
        h.strength = 150.0f + distReal(rng) * 50.0f; // 100-200
        h.radius = 0.15f;
        h.is_deep = true;
        hotspots.push_back(h);
    }
    
    // Shallow plumes (Hawaii-style): smaller radius, moderate strength, oceanic only
    for (int i = 0; i < params.shallow_plume_freq; ++i) {
        Vector3 pos(distReal(rng), distReal(rng), distReal(rng));
        Hotspot h;
        h.position = Math::normalize(pos);
        h.strength = 80.0f + distReal(rng) * 40.0f; // 40-120
        h.radius = 0.08f;
        h.is_deep = false;
        hotspots.push_back(h);
    }
}

void TectonicSimulator::simulateHotspots() {
    // Actual hotspot simulation is in CUDA
}

void TectonicSimulator::simulateStep(const SimulationParameters& params) {
    // Actual simulateStep is in CUDA
}

void TectonicSimulator::simulate(int iterations, const SimulationParameters& params) {
    generateHotspots(params);

    auto& cells = planet.getCells();
    int num_cells = cells.size();

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_elev(num_cells), h_vel_x(num_cells), h_vel_y(num_cells), h_vel_z(num_cells);
    std::vector<int> h_plate_id(num_cells);
    std::vector<uint8_t> h_is_oceanic(num_cells);
    std::vector<float> h_crustal_age(num_cells), h_crustal_thickness(num_cells);
    std::vector<int> h_neighbors(num_cells * 6, -1);
    std::vector<int> h_num_neighbors(num_cells, 0);

    for (int i = 0; i < num_cells; ++i) {
        h_pos_x[i] = cells[i].position.x;
        h_pos_y[i] = cells[i].position.y;
        h_pos_z[i] = cells[i].position.z;
        h_elev[i] = cells[i].elevation;
        h_vel_x[i] = cells[i].plate_velocity.x;
        h_vel_y[i] = cells[i].plate_velocity.y;
        h_vel_z[i] = cells[i].plate_velocity.z;
        h_plate_id[i] = cells[i].plate_id;
        h_is_oceanic[i] = cells[i].is_oceanic ? 1 : 0;
        h_crustal_age[i] = cells[i].crustal_age;
        h_crustal_thickness[i] = cells[i].crustal_thickness;

        h_num_neighbors[i] = cells[i].neighbors.size();
        for (size_t n = 0; n < cells[i].neighbors.size(); ++n) {
            h_neighbors[i * 6 + n] = cells[i].neighbors[n];
        }
    }

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_elev, *d_next_elev;
    float *d_vel_x, *d_vel_y, *d_vel_z, *d_age, *d_thick;
    int *d_plate_id, *d_neighbors, *d_num_neighbors;
    uint8_t *d_is_oceanic;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_age, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_thick, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_id, num_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_is_oceanic, num_cells * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

    int numPlates = plates.size();
    float *d_plate_axes_x, *d_plate_axes_y, *d_plate_axes_z, *d_plate_speeds;
    float *d_plate_centers_x, *d_plate_centers_y, *d_plate_centers_z;
    CHECK_CUDA(cudaMalloc(&d_plate_axes_x, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_axes_y, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_axes_z, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_speeds, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_x, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_y, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_z, numPlates * sizeof(float)));

    float *d_next_age, *d_next_thick;
    float *d_next_vx, *d_next_vy, *d_next_vz;

    CHECK_CUDA(cudaMalloc(&d_next_age, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_thick, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_vx, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_vy, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_vz, num_cells * sizeof(float)));

    std::vector<float> h_plate_axes_x(numPlates), h_plate_axes_y(numPlates), h_plate_axes_z(numPlates);
    std::vector<float> h_plate_speeds(numPlates);
    std::vector<float> h_plate_centers_x(numPlates), h_plate_centers_y(numPlates), h_plate_centers_z(numPlates);
    for (int p = 0; p < numPlates; ++p) {
        h_plate_axes_x[p] = plates[p].rotation_axis.x;
        h_plate_axes_y[p] = plates[p].rotation_axis.y;
        h_plate_axes_z[p] = plates[p].rotation_axis.z;
        h_plate_speeds[p] = plates[p].angular_speed;
        h_plate_centers_x[p] = plates[p].center.x;
        h_plate_centers_y[p] = plates[p].center.y;
        h_plate_centers_z[p] = plates[p].center.z;
    }
    CHECK_CUDA(cudaMemcpy(d_plate_axes_x, h_plate_axes_x.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_axes_y, h_plate_axes_y.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_axes_z, h_plate_axes_z.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_speeds, h_plate_speeds.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_x, h_plate_centers_x.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_y, h_plate_centers_y.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_z, h_plate_centers_z.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_elev, h_elev.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel_x, h_vel_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel_y, h_vel_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vel_z, h_vel_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_age, h_crustal_age.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_thick, h_crustal_thickness.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_id, h_plate_id.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_is_oceanic, h_is_oceanic.data(), num_cells * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int num_hotspots = hotspots.size();
    float *d_hs_x, *d_hs_y, *d_hs_z, *d_hs_str, *d_hs_radius;
    if (num_hotspots > 0) {
        CHECK_CUDA(cudaMalloc(&d_hs_x, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_y, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_z, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_str, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_radius, num_hotspots * sizeof(float)));
    }

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    float *d_stress, *d_next_stress;
    CHECK_CUDA(cudaMalloc(&d_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_stress, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_next_stress, 0, num_cells * sizeof(float)));

    if (params.use_bisector_distance) {
        // Use Exact Mathematical Distance to Bisector
        tectonic_bisector_stress_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_stress, d_plate_id, numPlates,
            d_plate_centers_x, d_plate_centers_y, d_plate_centers_z,
            d_plate_axes_x, d_plate_axes_y, d_plate_axes_z, d_plate_speeds
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // Calculate initial stress at boundaries
        tectonic_calc_stress_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_stress, d_plate_id, d_vel_x, d_vel_y, d_vel_z,
            d_neighbors, d_num_neighbors
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Diffuse stress inland to create smooth falloff (20 iterations)
        for (int i = 0; i < 20; ++i) {
            tectonic_diffuse_stress_kernel<<<numBlocks, blockSize>>>(
                num_cells, d_stress, d_next_stress, d_neighbors, d_num_neighbors
            );
            CHECK_CUDA(cudaDeviceSynchronize());
            std::swap(d_stress, d_next_stress);
        }
    }

    // Apply the smoothed stress to elevation with smoothstep
    tectonic_apply_stress_kernel<<<numBlocks, blockSize>>>(
        num_cells, d_elev, d_stress, d_is_oceanic, params.orogenesis_factor
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    if (num_hotspots > 0) {
        std::vector<float> h_hs_x(num_hotspots), h_hs_y(num_hotspots), h_hs_z(num_hotspots);
        std::vector<float> h_hs_str(num_hotspots), h_hs_rad(num_hotspots);
        for (int h = 0; h < num_hotspots; ++h) {
            h_hs_x[h] = hotspots[h].position.x;
            h_hs_y[h] = hotspots[h].position.y;
            h_hs_z[h] = hotspots[h].position.z;
            h_hs_str[h] = hotspots[h].strength;
            h_hs_rad[h] = hotspots[h].radius;
        }
        CHECK_CUDA(cudaMemcpy(d_hs_x, h_hs_x.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_y, h_hs_y.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_z, h_hs_z.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_str, h_hs_str.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_radius, h_hs_rad.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));

        tectonic_hotspot_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_elev, d_is_oceanic,
            num_hotspots, d_hs_x, d_hs_y, d_hs_z, d_hs_str, d_hs_radius
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Apply final high-quality global FBM noise with per-plate offsets
    tectonic_fbm_kernel<<<numBlocks, blockSize>>>(
        num_cells, d_pos_x, d_pos_y, d_pos_z,
        d_elev, d_stress, d_is_oceanic, d_plate_id
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Edge smoothing: Blend the cliffs created by per-plate offsets at boundaries
    for (int s = 0; s < 5; ++s) { // 5 passes of global smoothing
        tectonic_smooth_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_elev, d_next_elev, nullptr, d_neighbors, d_num_neighbors // Pass nullptr to ignore plate boundaries and smooth everything
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(d_elev, d_next_elev);
    }

    CHECK_CUDA(cudaMemcpy(h_elev.data(), d_elev, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_crustal_thickness.data(), d_thick, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_plate_id.data(), d_plate_id, num_cells * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_is_oceanic.data(), d_is_oceanic, num_cells * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_crustal_age.data(), d_age, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vel_x.data(), d_vel_x, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vel_y.data(), d_vel_y, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vel_z.data(), d_vel_z, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cells; ++i) {
        cells[i].elevation = h_elev[i];
        cells[i].crustal_thickness = h_crustal_thickness[i];
        cells[i].plate_id = h_plate_id[i];
        cells[i].is_oceanic = (h_is_oceanic[i] == 1);
        cells[i].crustal_age = h_crustal_age[i];
        cells[i].plate_velocity = Vector3(h_vel_x[i], h_vel_y[i], h_vel_z[i]);
        if (cells[i].elevation > 5000.0f) { 
            cells[i].bedrock = RockType::METAMORPHIC;
        }
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_elev); cudaFree(d_next_elev);
    cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
    cudaFree(d_age); cudaFree(d_thick);
    cudaFree(d_plate_id); cudaFree(d_is_oceanic);
    cudaFree(d_next_age); cudaFree(d_next_thick);
    cudaFree(d_next_vx); cudaFree(d_next_vy); cudaFree(d_next_vz);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
    cudaFree(d_plate_axes_x); cudaFree(d_plate_axes_y); cudaFree(d_plate_axes_z);
    cudaFree(d_plate_speeds);
    cudaFree(d_stress); cudaFree(d_next_stress);

    if (num_hotspots > 0) {
        cudaFree(d_hs_x); cudaFree(d_hs_y); cudaFree(d_hs_z); cudaFree(d_hs_str); cudaFree(d_hs_radius);
    }
}

} // namespace Ravis
