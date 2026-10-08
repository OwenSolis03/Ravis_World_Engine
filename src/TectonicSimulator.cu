#include "../include/TectonicSimulator.h"
#include "../include/MathUtils.h"
#include "../include/Noise.h"
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
// 3D noise — thin wrappers over the shared coherent value noise in Noise.h.
// The old sin-dot hash produced hexagonal / banded lattice artifacts on the
// sphere; value noise is direction-free and spatially smooth. Device and host
// share one implementation, so tectonics and climate stay in sync by
// construction.
// ============================================================================
__device__ float hash_noise3d(float x, float y, float z) {
    return rw_value_noise3d(x, y, z);
}

__device__ float hash_fbm3d(float x, float y, float z, int octaves) {
    return rw_fbm3d(x, y, z, octaves);
}

__device__ float hash_ridge3d(float x, float y, float z, int octaves) {
    return rw_ridge3d(x, y, z, octaves);
}

static float cpu_hash_noise3d(float x, float y, float z) {
    return rw_value_noise3d(x, y, z);
}

static float cpu_hash_fbm3d(float x, float y, float z, int octaves) {
    return rw_fbm3d(x, y, z, octaves);
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
        // Smoothstep falloff mapping. mag MUST be clamped to [0,1] first: stress
        // routinely reaches 3-5 at plate boundaries, and for mag > 1.5 the cubic
        // (3 - 2*mag) turns negative and unbounded -> +/- hundreds of km of uplift
        // that then feed NaNs/inf into the erosion pass.
        float mag = fminf(fabsf(s), 1.0f);
        float falloff = mag * mag * (3.0f - 2.0f * mag); // Smooth curve, now in [0,1]
        
        float delta = 0.0f;
        if (s > 0.0f) { // Convergence (Mountains / Subduction)
            if (oceanic) delta = -falloff * 8000.0f * orogenesis_factor; // Oceanic Trench (Subduction)
            else delta = falloff * 12000.0f * orogenesis_factor; // Continental Mountains (Himalayas/Andes)
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
    float stress_multiplier = min(1.0f, s * 0.35f);

    float current_elev = elevation[i];

    if (!is_oceanic[i]) {
        // Continental terrain
        current_elev += low_noise * 750.0f; // Base continent shape (plateaus)

        // Ridge structure: a base amount everywhere (ancient eroded ranges,
        // hill country) plus a much larger stress-scaled component at active
        // convergent boundaries.
        current_elev += ridge_hills  * (450.0f + 3500.0f * stress_multiplier);
        current_elev += ridge_detail * (180.0f + 1800.0f * stress_multiplier);
        
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
            // Continental plates: bias seed placement toward clustering poles.
            // Scaled down: with few continental plates the raw slider value
            // fused every continent into one supercontinent.
            float cluster_strength = params.primary_clustering * 0.6f;
            float secondary_strength = params.secondary_clustering * 0.6f;
            
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

    // Assign every cell to the nearest plate seed, measured from a
    // noise-warped position: continuous-space Voronoi with domain warping.
    // Boundaries come out as smooth organic curves with no lattice/hexagon
    // artifacts (BFS / region-growing on the triangular cell graph produces
    // hexagons no matter how it's jittered). crust_warping controls how far
    // the boundaries wander from a clean Voronoi diagram.
    auto assign_warped_voronoi = [&](int nseeds) {
        const float wa = 0.06f + 0.30f * params.crust_warping; // warp amplitude
        for (auto& cell : cells) {
            float px = static_cast<float>(cell.position.x);
            float py = static_cast<float>(cell.position.y);
            float pz = static_cast<float>(cell.position.z);
            float wx = px + wa * cpu_hash_fbm3d(px * 2.4f + 11.3f, py * 2.4f,         pz * 2.4f,         4);
            float wy = py + wa * cpu_hash_fbm3d(px * 2.4f,         py * 2.4f + 24.7f,  pz * 2.4f,         4);
            float wz = pz + wa * cpu_hash_fbm3d(px * 2.4f,         py * 2.4f,          pz * 2.4f + 39.1f, 4);
            int best = 0;
            float best_d = 1e30f;
            for (int p = 0; p < nseeds; ++p) {
                float dx = wx - static_cast<float>(plates[p].center.x);
                float dy = wy - static_cast<float>(plates[p].center.y);
                float dz = wz - static_cast<float>(plates[p].center.z);
                float d = dx * dx + dy * dy + dz * dz;
                if (d < best_d) { best_d = d; best = p; }
            }
            cell.plate_id = best;
        }
    };

    // Major plates first, so minor-plate seeds land on real boundaries.
    assign_warped_voronoi(numMajor);

    // Find boundaries of the major plates
    std::vector<size_t> boundary_cells;
    for (size_t i = 0; i < cells.size(); ++i) {
        int pid = cells[i].plate_id;
        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].plate_id != pid) { boundary_cells.push_back(i); break; }
        }
    }

    // Seed minor plates (fast-moving fragments) on the boundaries, then do the
    // final assignment over every plate.
    if (!boundary_cells.empty()) {
        std::uniform_int_distribution<size_t> distBoundary(0, boundary_cells.size() - 1);
        for (int i = numMajor; i < numPlates; ++i) {
            size_t seedId = boundary_cells[distBoundary(rng)];
            plates[i].is_oceanic = distProb(rng) > params.crust_fraction;
            plates[i].center = cells[seedId].position;
            Vector3 axis(distReal(rng), distReal(rng), distReal(rng));
            plates[i].rotation_axis = Math::normalize(axis);
            if (plates[i].is_oceanic) {
                plates[i].angular_speed = distProb(rng) * speed_scale * 0.3f + 0.01f;
            } else {
                plates[i].angular_speed = distProb(rng) * speed_scale * 0.15f + 0.005f;
            }
        }
        assign_warped_voronoi(numPlates);
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
            ridge = std::pow(ridge, 1.9f);

            cell.elevation = 50.0f + ridge * 3000.0f + distContVar(rng);
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

// Adds the discrete "terrain feature" knobs (superswells, ancient ranges,
// hills, uplands, stochastic uplifts) on top of the tectonic base. Runs after
// simulate() so it shapes the final terrain; erosion then works it down.
// Each feature is a noise-warped radial dome, not a BFS flood fill — flood
// fill on the triangular cell graph produces hexagons.
void TectonicSimulator::applyTerrainFeatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    if (cells.empty()) return;

    std::mt19937 rng(params.seed + 7777);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);

    // ang_radius: chord radius on the unit sphere (0.10 ~ 6 deg, 0.55 ~ 33 deg).
    auto add_dome = [&](float ang_radius, float peak, float rim, bool land_only) {
        Vector3 c = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        float phase = distProb(rng) * 100.0f;
        for (auto& cell : cells) {
            if (land_only && cell.is_oceanic) continue;
            float px = static_cast<float>(cell.position.x);
            float py = static_cast<float>(cell.position.y);
            float pz = static_cast<float>(cell.position.z);
            float dx = px - static_cast<float>(c.x);
            float dy = py - static_cast<float>(c.y);
            float dz = pz - static_cast<float>(c.z);
            float chord = std::sqrt(dx * dx + dy * dy + dz * dz);

            // Irregular outline: wobble the effective radius with low-freq noise.
            float wob = cpu_hash_fbm3d(px * 4.0f + phase, py * 4.0f, pz * 4.0f, 3); // [-1,1]
            float r = ang_radius * (1.0f + 0.4f * wob);
            if (chord >= r) continue;

            float t = chord / r;                      // 0 centre .. 1 edge
            float falloff = (1.0f - t) * (1.0f - t);  // smooth dome
            float uplift = peak * falloff + rim * (1.0f - falloff);
            uplift *= 0.75f + 0.25f * std::abs(cpu_hash_noise3d(px * 13.0f, py * 13.0f, pz * 13.0f));

            cell.elevation += uplift;
            if (!cell.is_oceanic) cell.crustal_thickness += uplift / 120.0f;
        }
    };

    // Linear range: uplift along a great-circle arc so it reads as a cordillera
    // rather than a circular blob. arc_len in radians, half_width chord units.
    auto add_ridge = [&](float arc_len, float half_width, float peak, float rim, bool land_only) {
        Vector3 a = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        Vector3 dir = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        // component of dir tangent to the sphere at a
        double adot = a.x * dir.x + a.y * dir.y + a.z * dir.z;
        Vector3 tang = Math::normalize(Vector3(dir.x - adot * a.x,
                                               dir.y - adot * a.y,
                                               dir.z - adot * a.z));
        float cs = std::cos(arc_len), sn = std::sin(arc_len);
        Vector3 b = Math::normalize(Vector3(a.x * cs + tang.x * sn,
                                            a.y * cs + tang.y * sn,
                                            a.z * cs + tang.z * sn));
        float abx = static_cast<float>(b.x - a.x);
        float aby = static_cast<float>(b.y - a.y);
        float abz = static_cast<float>(b.z - a.z);
        float ab2 = abx * abx + aby * aby + abz * abz + 1e-9f;
        float phase = distProb(rng) * 100.0f;

        for (auto& cell : cells) {
            if (land_only && cell.is_oceanic) continue;
            float px = static_cast<float>(cell.position.x);
            float py = static_cast<float>(cell.position.y);
            float pz = static_cast<float>(cell.position.z);
            // distance to the chord segment a..b
            float apx = px - static_cast<float>(a.x);
            float apy = py - static_cast<float>(a.y);
            float apz = pz - static_cast<float>(a.z);
            float u = (apx * abx + apy * aby + apz * abz) / ab2;
            u = std::max(0.0f, std::min(1.0f, u));
            float dx = apx - abx * u, dy = apy - aby * u, dz = apz - abz * u;
            float d = std::sqrt(dx * dx + dy * dy + dz * dz);

            float wob = cpu_hash_fbm3d(px * 5.0f + phase, py * 5.0f, pz * 5.0f, 3); // [-1,1]
            float w = half_width * (1.0f + 0.5f * wob);
            if (d >= w) continue;

            float t = d / w;
            float falloff = (1.0f - t) * (1.0f - t);
            float uplift = peak * falloff + rim * (1.0f - falloff);
            uplift *= 0.7f + 0.3f * std::abs(cpu_hash_noise3d(px * 14.0f, py * 14.0f, pz * 14.0f));

            cell.elevation += uplift;
            if (!cell.is_oceanic) cell.crustal_thickness += uplift / 120.0f;
        }
    };

    // Superswells (African-style): broad, gentle, also lifts ocean floor.
    for (int i = 0; i < params.superswell_freq; ++i)  add_dome(0.55f,  900.0f,   0.0f, false);
    // Old mountains (Appalachian/Ural-style): long inland cordilleras.
    for (int i = 0; i < params.old_mountain_freq; ++i) add_ridge(0.60f, 0.075f, 2600.0f, 300.0f, true);
    // Old hills (Caledonian-style): shorter, lower linear ridges.
    for (int i = 0; i < params.old_hill_freq; ++i)     add_ridge(0.40f, 0.060f,  950.0f, 120.0f, true);
    // Small stochastic uplifts.
    for (int i = 0; i < params.small_uplift_freq; ++i) add_dome(0.09f,  500.0f,  80.0f, true);
    // Uplands (Volga-style): very broad, very gentle.
    for (int i = 0; i < params.upland_freq; ++i)       add_dome(0.40f,  420.0f, 120.0f, true);
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

        // Diffuse stress inland to create smooth falloff. More passes = uplift
        // reaches further from the plate boundary, so mountain belts have broad
        // flanks / foothills instead of a thin coastal ridge.
        for (int i = 0; i < 40; ++i) {
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
        float e = h_elev[i];
        if (!std::isfinite(e)) e = -3500.0f; // NaN/inf guard: fall back to ocean floor
        cells[i].elevation = e;
        cells[i].crustal_thickness = h_crustal_thickness[i];
        cells[i].plate_id = h_plate_id[i];
        cells[i].is_oceanic = (h_is_oceanic[i] == 1);
        cells[i].crustal_age = h_crustal_age[i];
        cells[i].plate_velocity = Vector3(h_vel_x[i], h_vel_y[i], h_vel_z[i]);
        if (cells[i].elevation > 5000.0f) {
            cells[i].bedrock = RockType::METAMORPHIC;
        }
    }

    // Remove landlocked deep basins: flood-fill ocean cells, keep the largest
    // connected body as the "world ocean"; any other ocean region (a small
    // oceanic plate that ended up enclosed by continents) is raised to a
    // shallow inland-sea depth so subdivided worlds don't show 4-10 km abysses
    // sitting inside a landmass.
    {
        const float sea = params.effective_sea_level();
        std::vector<int> comp(num_cells, -1);
        std::vector<int> compSize;
        std::queue<int> fq;
        for (int i = 0; i < num_cells; ++i) {
            if (cells[i].elevation > sea || comp[i] != -1) continue;
            int id = static_cast<int>(compSize.size());
            int sz = 0;
            comp[i] = id;
            fq.push(i);
            while (!fq.empty()) {
                int c = fq.front(); fq.pop();
                ++sz;
                for (size_t nid : cells[c].neighbors) {
                    if (cells[nid].elevation <= sea && comp[nid] == -1) {
                        comp[nid] = id;
                        fq.push(static_cast<int>(nid));
                    }
                }
            }
            compSize.push_back(sz);
        }
        if (compSize.size() > 1) {
            int mainComp = 0;
            for (int c = 1; c < static_cast<int>(compSize.size()); ++c)
                if (compSize[c] > compSize[mainComp]) mainComp = c;
            // Everything that is not the one world ocean becomes a shallow
            // inland sea rather than a deep basin.
            const float inland_floor = sea - 60.0f;
            for (int i = 0; i < num_cells; ++i) {
                if (comp[i] != -1 && comp[i] != mainComp && cells[i].elevation < inland_floor)
                    cells[i].elevation = inland_floor;
            }
        }
    }

    // Discrete terrain-feature knobs (superswells, ancient ranges/hills,
    // uplands, stochastic uplifts) applied on top of the tectonic base.
    applyTerrainFeatures(params);

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
