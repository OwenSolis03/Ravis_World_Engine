#include "../include/TectonicSimulator.h"
#include "../include/MathUtils.h"
#include "../include/Noise.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

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
// Wrappers de ruido 3D coherente con soporte de semilla (Noise.h)
// ============================================================================
__device__ inline float hash_noise3d(float x, float y, float z, uint32_t seed) {
    return rw_value_noise3d(x, y, z, seed);
}

__device__ inline float hash_fbm3d(float x, float y, float z, int octaves, uint32_t seed) {
    return rw_fbm3d(x, y, z, octaves, seed);
}

__device__ inline float hash_ridge3d(float x, float y, float z, int octaves, uint32_t seed) {
    return rw_ridge3d(x, y, z, octaves, seed);
}

static inline float cpu_hash_noise3d(float x, float y, float z, uint32_t seed) {
    return rw_value_noise3d(x, y, z, seed);
}

static inline float cpu_hash_fbm3d(float x, float y, float z, int octaves, uint32_t seed) {
    return rw_fbm3d(x, y, z, octaves, seed);
}

static inline float cpu_hash_ridge3d(float x, float y, float z, int octaves, uint32_t seed) {
    return rw_ridge3d(x, y, z, octaves, seed);
}

// ============================================================================
// Kernels CUDA Geofísicos
// ============================================================================

// Evalúa la cinemática en los bordes de placa y marca `is_collision = 1` cuando
// chocan dos bloques continentales (sutura interior tipo Himalaya) vs `0` cuando es
// subducción costera (Oceánica-Continental) o arco oceánico (Oceánica-Oceánica).
__global__ void tectonic_boundary_kinematics_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const int* plate_id,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const uint8_t* is_oceanic,
    const float* crustal_age,
    float* comp_stress,
    float* div_stress,
    float* shear_stress,
    uint8_t* is_collision,
    int* dist_to_boundary,
    const int* neighbors, const int* num_neighbors,
    float speed_norm_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float vx = vel_x[i], vy = vel_y[i], vz = vel_z[i];
    bool my_oceanic = (is_oceanic[i] != 0);
    float my_age = crustal_age[i];

    float best_comp = 0.0f;
    float best_div = 0.0f;
    float best_shear = 0.0f;
    uint8_t coll_flag = 0;
    bool on_boundary = false;

    int nn = num_neighbors[i];
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        int npid = plate_id[nid];
        if (npid == pid) continue;

        on_boundary = true;

        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        dx /= dist; dy /= dist; dz /= dist;

        float rvx = vx - vel_x[nid];
        float rvy = vy - vel_y[nid];
        float rvz = vz - vel_z[nid];

        float v_conv = (rvx * dx + rvy * dy + rvz * dz) * speed_norm_inv;

        float tx = rvx * speed_norm_inv - v_conv * dx;
        float ty = rvy * speed_norm_inv - v_conv * dy;
        float tz = rvz * speed_norm_inv - v_conv * dz;
        float v_shear = sqrtf(tx * tx + ty * ty + tz * tz);
        best_shear = fmaxf(best_shear, fminf(1.5f, v_shear));

        bool nbr_oceanic = (is_oceanic[nid] != 0);
        float nbr_age = crustal_age[nid];

        if (v_conv > 0.0f) {
            float conv_mag = fminf(2.0f, v_conv * 1.35f);
            float signed_orogeny = 0.0f;
            uint8_t local_coll = 0;

            if (my_oceanic && !nbr_oceanic) {
                signed_orogeny = -conv_mag * 1.15f; // Fosa de subducción
            } else if (!my_oceanic && nbr_oceanic) {
                signed_orogeny = conv_mag * 0.90f;  // Subducción costera (moderada)
                local_coll = 0;
            } else if (my_oceanic && nbr_oceanic) {
                if (my_age > nbr_age + 2.0f || (fabsf(my_age - nbr_age) <= 2.0f && pid > npid)) {
                    signed_orogeny = -conv_mag * 1.05f;
                } else {
                    signed_orogeny = conv_mag * 1.10f; // Arco insular oceánico (Indonesia/Filipinas)
                }
            } else {
                // Sutura Continental-Continental en el interior de los continentes (Himalaya/Urales)
                signed_orogeny = conv_mag * 1.65f;
                local_coll = 1;
            }

            if (fabsf(signed_orogeny) > fabsf(best_comp)) {
                best_comp = signed_orogeny;
                coll_flag = local_coll;
            }
        } else if (v_conv < 0.0f) {
            float div_mag = fminf(1.8f, (-v_conv) * 1.35f);
            best_div = fmaxf(best_div, div_mag);
        }
    }

    comp_stress[i] = best_comp;
    div_stress[i] = best_div;
    shear_stress[i] = best_shear;
    is_collision[i] = coll_flag;
    dist_to_boundary[i] = on_boundary ? 0 : 9999;
}

__global__ void tectonic_bisector_stress_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const int* plate_id, const uint8_t* is_oceanic, const float* crustal_age,
    float* comp_stress, float* div_stress, float* shear_stress,
    uint8_t* is_collision, int* dist_to_boundary,
    int numPlates,
    const float* center_x, const float* center_y, const float* center_z,
    const float* axis_x, const float* axis_y, const float* axis_z,
    const float* speeds, const uint8_t* plate_is_oceanic,
    float speed_norm_inv, int res_mult)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];

    float c1x = center_x[pid], c1y = center_y[pid], c1z = center_z[pid];
    float dx1 = px - c1x, dy1 = py - c1y, dz1 = pz - c1z;
    float dist1 = sqrtf(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + 1e-6f);

    float min_bisector_dist = 1e10f;
    int closest_pid = -1;

    for (int p = 0; p < numPlates; ++p) {
        if (p == pid) continue;
        float c2x = center_x[p], c2y = center_y[p], c2z = center_z[p];
        float dx2 = px - c2x, dy2 = py - c2y, dz2 = pz - c2z;
        float dist2 = sqrtf(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + 1e-6f);

        float b_dist = fmaxf(0.0f, (dist2 - dist1) * 0.5f);
        if (b_dist < min_bisector_dist) {
            min_bisector_dist = b_dist;
            closest_pid = p;
        }
    }

    if (closest_pid != -1) {
        float v1x = (axis_y[pid] * pz - axis_z[pid] * py) * speeds[pid];
        float v1y = (axis_z[pid] * px - axis_x[pid] * pz) * speeds[pid];
        float v1z = (axis_x[pid] * py - axis_y[pid] * px) * speeds[pid];

        float v2x = (axis_y[closest_pid] * pz - axis_z[closest_pid] * py) * speeds[closest_pid];
        float v2y = (axis_z[closest_pid] * px - axis_x[closest_pid] * pz) * speeds[closest_pid];
        float v2z = (axis_x[closest_pid] * py - axis_y[closest_pid] * px) * speeds[closest_pid];

        float nx = center_x[closest_pid] - c1x;
        float ny = center_y[closest_pid] - c1y;
        float nz = center_z[closest_pid] - c1z;
        float nlen = sqrtf(nx * nx + ny * ny + nz * nz + 1e-6f);
        nx /= nlen; ny /= nlen; nz /= nlen;

        float rvx = (v1x - v2x) * speed_norm_inv;
        float rvy = (v1y - v2y) * speed_norm_inv;
        float rvz = (v1z - v2z) * speed_norm_inv;

        float v_conv = rvx * nx + rvy * ny + rvz * nz;
        float tx = rvx - v_conv * nx, ty = rvy - v_conv * ny, tz = rvz - v_conv * nz;
        float v_shear = sqrtf(tx * tx + ty * ty + tz * tz);

        float falloff = fmaxf(0.0f, 1.0f - (min_bisector_dist * 7.5f));
        bool my_oceanic = (is_oceanic[i] != 0);
        bool other_oceanic = (plate_is_oceanic[closest_pid] != 0);

        is_collision[i] = 0;
        if (v_conv > 0.0f) {
            float conv_mag = fminf(2.0f, v_conv * 1.35f) * falloff;
            if (my_oceanic && !other_oceanic) {
                comp_stress[i] = -conv_mag * 1.15f;
            } else if (!my_oceanic && other_oceanic) {
                comp_stress[i] = conv_mag * 0.90f;
            } else if (my_oceanic && other_oceanic) {
                comp_stress[i] = (pid > closest_pid) ? (-conv_mag * 0.9f) : (conv_mag * 1.05f);
            } else {
                comp_stress[i] = conv_mag * 1.65f;
                is_collision[i] = 1;
            }
            div_stress[i] = 0.0f;
        } else {
            comp_stress[i] = 0.0f;
            div_stress[i] = fminf(1.8f, (-v_conv) * 1.35f) * falloff;
        }
        shear_stress[i] = fminf(1.5f, v_shear) * falloff;
        dist_to_boundary[i] = static_cast<int>(min_bisector_dist * 32.0f * static_cast<float>(res_mult));
    } else {
        comp_stress[i] = 0.0f;
        div_stress[i] = 0.0f;
        shear_stress[i] = 0.0f;
        is_collision[i] = 0;
        dist_to_boundary[i] = 9999;
    }
}

__global__ void tectonic_propagate_stress_kernel(
    int num_cells,
    const int* plate_id,
    const float* comp_in, float* comp_out,
    const float* div_in, float* div_out,
    const float* shear_in, float* shear_out,
    const uint8_t* coll_in, uint8_t* coll_out,
    const int* dist_in, int* dist_out,
    const int* neighbors, const int* num_neighbors,
    float step_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int pid = plate_id[i];
    int nn = num_neighbors[i];

    float sum_comp = comp_in[i] * 1.5f;
    float sum_div = div_in[i] * 1.5f;
    float sum_shear = shear_in[i] * 1.5f;
    float weight_same = 1.5f;

    float max_abs_comp = fabsf(comp_in[i]);
    float max_div = div_in[i];
    uint8_t coll_flag = coll_in[i];
    int min_dist = dist_in[i];

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        if (dist_in[nid] + 1 < min_dist) {
            min_dist = dist_in[nid] + 1;
            if (plate_id[nid] == pid) {
                coll_flag = coll_in[nid];
            }
        }
        if (plate_id[nid] == pid) {
            float nc = comp_in[nid];
            float nd = div_in[nid];
            float ns = shear_in[nid];
            sum_comp += nc;
            sum_div += nd;
            sum_shear += ns;
            weight_same += 1.0f;

            if (fabsf(nc) * step_decay > max_abs_comp) {
                max_abs_comp = fabsf(nc) * step_decay;
                coll_flag = coll_in[nid];
                if ((nc < 0.0f && sum_comp <= 0.0f) || (nc > 0.0f && sum_comp >= 0.0f)) {
                    sum_comp += 0.35f * nc;
                    weight_same += 0.35f;
                }
            }
            if (nd * step_decay > max_div) {
                max_div = nd * step_decay;
            }
        }
    }

    float avg_comp = sum_comp / weight_same;
    float sign_c = (avg_comp >= 0.0f) ? 1.0f : -1.0f;
    comp_out[i] = sign_c * fmaxf(fabsf(avg_comp), max_abs_comp * step_decay);
    div_out[i] = fmaxf(sum_div / weight_same, max_div * step_decay);
    shear_out[i] = sum_shear / weight_same;
    coll_out[i] = coll_flag;
    dist_out[i] = min_dist;
}

__global__ void tectonic_advect_and_age_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const int* plate_id,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const uint8_t* is_oceanic_in, uint8_t* is_oceanic_out,
    const float* age_in, float* age_out,
    const float* thick_in, float* thick_out,
    const float* div_stress,
    const int* dist_to_boundary,
    const int* neighbors, const int* num_neighbors,
    float dt_Myr, float step_rad_scale, float hop_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float vx = vel_x[i] * step_rad_scale;
    float vy = vel_y[i] * step_rad_scale;
    float vz = vel_z[i] * step_rad_scale;

    float tx = px - vx;
    float ty = py - vy;
    float tz = pz - vz;
    float inv_len = rsqrtf(tx * tx + ty * ty + tz * tz + 1e-12f);
    tx *= inv_len; ty *= inv_len; tz *= inv_len;

    int best_j = i;
    float best_dot = px * tx + py * ty + pz * tz;
    int pid = plate_id[i];

    for (int hop = 0; hop < 3; ++hop) {
        bool moved = false;
        int nn = num_neighbors[best_j];
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[best_j * 6 + n];
            if (plate_id[nid] != pid) continue;
            float d = pos_x[nid] * tx + pos_y[nid] * ty + pos_z[nid] * tz;
            if (d > best_dot) {
                best_dot = d;
                best_j = nid;
                moved = true;
            }
        }
        if (!moved) break;
    }

    uint8_t oceanic = is_oceanic_in[best_j];
    float age = 0.75f * age_in[i] + 0.25f * age_in[best_j] + dt_Myr;
    float thick = 0.75f * thick_in[i] + 0.25f * thick_in[best_j];

    float div = div_stress[i];
    float d_hops = static_cast<float>(dist_to_boundary[i]) * hop_scale;
    if (div > 0.15f && d_hops <= 3.0f) {
        float ridge_factor = fminf(1.0f, div) * (1.0f - d_hops * 0.25f);
        if (oceanic) {
            age = age * (1.0f - 0.65f * ridge_factor);
            thick = thick * (1.0f - 0.30f * ridge_factor) + 6.0f * (0.30f * ridge_factor);
        } else if (div > 0.55f && d_hops <= 1.0f) {
            thick = fmaxf(18.0f, thick - 1.2f * ridge_factor);
        }
    }

    is_oceanic_out[i] = oceanic;
    age_out[i] = fminf(oceanic ? 280.0f : 4000.0f, age);
    thick_out[i] = thick;
}

// Síntesis morfológica:
// 1. En colisiones Oceánica-Oceánica: crea complejos de archipiélagos tipo Indonesia/Singapur/Filipinas
//    (combinando plataformas anchas estilo Sonda con islas como Borneo/Java y cadenas de islas volcánicas separadas por estrechos).
// 2. En subducciones costeras (Oceánica-Continental): deja una llanura costera baja, desplaza el pico tierra adentro
//    y mantiene alturas moderadas (~2200m - 4200m) con pasos de montaña para no crear muros de 8km junto al mar.
// 3. En suturas intracontinentales (Continental-Continental): levanta grandes cordilleras interiores (5500m - 8500m).
__global__ void tectonic_synthesize_orogeny_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation,
    float* crustal_thickness,
    const uint8_t* is_oceanic,
    const float* crustal_age,
    const float* comp_stress,
    const float* div_stress,
    const float* shear_stress,
    const uint8_t* is_collision,
    const int* dist_to_boundary,
    const int* neighbors, const int* num_neighbors,
    float orogenesis_factor,
    float subsidence_rate,
    uint32_t seed,
    float hop_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    bool oceanic = (is_oceanic[i] != 0);
    float age = fmaxf(0.0f, crustal_age[i]);
    float comp = comp_stress[i];
    float div = div_stress[i];
    float shear = shear_stress[i];
    bool cont_suture = (is_collision[i] != 0);
    int bdist = dist_to_boundary[i];

    // Perturbación sinuosa de la distancia al borde para que las cordilleras y arcos de islas
    // zigzagueen orgánicamente en lugar de seguir una línea paralela perfecta a la falla.
    float meander = hash_fbm3d(px * 9.5f, py * 9.5f, pz * 9.5f, 3, seed + 41u) * 0.45f;
    float d_hops = fmaxf(0.0f, static_cast<float>(bdist) * hop_scale * 2.5f + meander);

    // Modulación a lo largo del rumbo con valles transversales (pasos de montaña)
    float strike_fbm = 0.5f + 0.5f * hash_fbm3d(px * 7.5f, py * 7.5f, pz * 7.5f, 4, seed + 19u);
    float pass_carver = hash_ridge3d(px * 14.0f, py * 14.0f, pz * 14.0f, 3, seed + 29u);
    float strike_mod = (0.35f + 0.95f * strike_fbm) * (0.45f + 0.65f * pass_carver);

    if (oceanic) {
        float depth = -2500.0f - subsidence_rate * sqrtf(fminf(age, 180.0f));
        float thick = 7.0f + 0.02f * sqrtf(fminf(age, 180.0f));

        // 1. Dorsal Meso-Oceánica en divergencia
        if (div > 0.01f) {
            float ridge_profile = expf(-d_hops * 0.45f) * fminf(1.2f, div);
            float axial_graben = (d_hops < 0.25f) ? -250.0f : 0.0f;
            depth = depth * (1.0f - ridge_profile) + (-2450.0f + axial_graben) * ridge_profile;
        }

        // 2. Subducción Oceánica (fosa) vs. Complejo de Archipiélagos (estilo Indonesia / Filipinas / Japón)
        if (comp < -0.01f) {
            float mag = fminf(1.5f, -comp);
            float trench_profile = expf(-d_hops * 0.85f);
            float outer_swell = 0.12f * expf(-0.5f * (d_hops - 2.5f) * (d_hops - 2.5f));
            depth += (-4400.0f * trench_profile + 320.0f * outer_swell) * mag * orogenesis_factor * strike_mod;
        } else if (comp > 0.01f) {
            float mag = fminf(1.5f, comp);

            // Envolvente regional de baja frecuencia: alterna zonas de plataforma ancha estilo Sonda
            // (Singapur / Sumatra / Borneo / Java) con zonas de arco volcánico insular fragmentado
            float sunda_Grouping = 0.5f + 0.5f * hash_fbm3d(px * 3.8f, py * 3.8f, pz * 3.8f, 3, seed + 61u);
            // Máscara de fragmentación de islas: crea estrechos marinos entre islas (< 0 = canal marino, > 0 = isla)
            float island_chain_mask = hash_fbm3d(px * 16.0f, py * 16.0f, pz * 16.0f, 4, seed + 67u);
            float volcanic_peaks = hash_ridge3d(px * 28.0f, py * 28.0f, pz * 28.0f, 4, seed + 73u);

            // a) Plataforma de retroarco ancha (Sunda Shelf): eleva el fondo abisal a un mar somero (-180m a +400m)
            //    de 2.5 a 4 unidades de ancho donde emergen grandes islas como Borneo o Sumatra
            float shelf_width = 1.4f + 1.8f * sunda_Grouping;
            float backarc_shelf = expf(-0.22f * (d_hops - 1.4f) * (d_hops - 1.4f) / (shelf_width * shelf_width));
            float sunda_factor = fmaxf(0.0f, (sunda_Grouping - 0.42f) * 1.72f);

            if (sunda_factor > 0.0f && backarc_shelf > 0.15f) {
                // Elevar el abismo hacia plataforma submarina somera (-120m) + masas de tierra anchas
                float target_sunda = -140.0f + (island_chain_mask + 0.18f) * 1150.0f + volcanic_peaks * 950.0f * strike_mod;
                float blend = fminf(1.0f, sunda_factor * backarc_shelf * mag * 1.15f);
                depth = depth * (1.0f - blend) + target_sunda * blend;
                thick += blend * 14.0f;
            }

            // b) Arco volcánico principal (cadenas de islas separadas por estrechos marinos en vez de una línea continua)
            float arc_profile = expf(-0.55f * (d_hops - 0.95f) * (d_hops - 0.95f));
            // Si `island_chain_mask` es negativo, es un estrecho marino entre islas; si es positivo, emerge la isla
            float island_emergence = fmaxf(-0.15f, island_chain_mask + 0.22f);
            float arc_uplift = (2200.0f + 3600.0f * island_emergence + 2100.0f * volcanic_peaks)
                               * arc_profile * mag * orogenesis_factor;
            depth += arc_uplift;
            thick += fmaxf(0.0f, arc_uplift * 0.003f);
        }

        // 3. Margen continental pasivo
        int nn = num_neighbors[i];
        int cont_nbrs = 0;
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[i * 6 + n];
            if (is_oceanic[nid] == 0) cont_nbrs++;
        }
        if (cont_nbrs > 0 && comp >= -0.25f) {
            float shelf_weight = fminf(1.0f, static_cast<float>(cont_nbrs) * 0.38f);
            float shelf_elev = -130.0f - 250.0f * (1.0f - shelf_weight);
            depth = depth * (1.0f - 0.75f * shelf_weight) + shelf_elev * (0.75f * shelf_weight);
        }

        elevation[i] = fmaxf(-11500.0f, depth);
        crustal_thickness[i] = thick;
    } else {
        // Corteza Continental
        float base_elev = elevation[i];
        float thick = crustal_thickness[i];

        if (comp > 0.01f) {
            float mag = fminf(1.6f, comp);

            if (cont_suture) {
                // COLISIÓN CONTINENTAL-CONTINENTAL (En el interior del continente, ej. Himalaya / Urales / Zagros):
                // Se permiten las mayores altitudes del planeta (5000m - 8200m) lejos del océano.
                float suture_core = expf(-0.35f * (d_hops - 0.6f) * (d_hops - 0.6f));
                float plateau_flank = 0.28f * expf(-0.08f * d_hops * d_hops);
                float orogenic_uplift = (4500.0f * suture_core + 1400.0f * plateau_flank)
                                        * mag * orogenesis_factor * strike_mod;
                base_elev += orogenic_uplift;
                thick += fmaxf(0.0f, orogenic_uplift * 0.0068f);
            } else {
                // SUBDUCCIÓN COSTERA (Oceánica bajo Continental, ej. Cascadia / Andes costeros):
                // Rampa costera suave (`coastal_taper`) para que la montaña NO suba de golpe en la playa,
                // pico desplazado tierra adentro (`d_hops ~ 1.65`) y altura moderada (~2200m - 4200m).
                float coastal_taper = fminf(1.0f, d_hops * 0.65f);
                float coastal_cordillera = expf(-0.48f * (d_hops - 1.65f) * (d_hops - 1.65f)) * coastal_taper;
                float inland_range = 0.35f * expf(-0.22f * (d_hops - 3.4f) * (d_hops - 3.4f)); // Segunda cadena interior (estilo Rocosas)
                float foreland_basin = -0.08f * expf(-0.25f * (d_hops - 5.2f) * (d_hops - 5.2f));

                float orogenic_uplift = (2400.0f * coastal_cordillera + 1500.0f * inland_range + 400.0f * foreland_basin)
                                        * mag * orogenesis_factor * strike_mod;
                base_elev += orogenic_uplift;
                thick += fmaxf(0.0f, orogenic_uplift * 0.0055f);
            }
        }

        if (div > 0.05f) {
            float rift_profile = expf(-0.60f * d_hops * d_hops) * fminf(1.3f, div);
            float rift_flank = 0.20f * expf(-0.8f * (d_hops - 1.2f) * (d_hops - 1.2f)) * fminf(1.3f, div);
            base_elev += (-1500.0f * rift_profile + 380.0f * rift_flank) * strike_mod;
            thick = fmaxf(20.0f, thick - 8.0f * rift_profile);
        }

        if (shear > 0.1f && d_hops <= 1.2f) {
            base_elev += 380.0f * shear * expf(-1.2f * d_hops) * strike_mod;
        }

        // Asegurar que las celdas inmediatamente en contacto con el océano tengan una transición costera realista
        int nn = num_neighbors[i];
        int ocean_nbrs = 0;
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[i * 6 + n];
            if (is_oceanic[nid] != 0) ocean_nbrs++;
        }
        if (ocean_nbrs > 0) {
            base_elev = fminf(base_elev, 450.0f) * 0.45f + 18.0f;
        }

        elevation[i] = fminf(8850.0f, base_elev);
        crustal_thickness[i] = fminf(75.0f, thick);
    }
}

__global__ void tectonic_hotspot_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation, const uint8_t* is_oceanic, const int* plate_id,
    const float* plate_ax_x, const float* plate_ax_y, const float* plate_ax_z,
    const float* plate_speed,
    int num_hotspots,
    const float* hs_x, const float* hs_y, const float* hs_z,
    const float* hs_str, const float* hs_radius, const uint8_t* hs_is_deep,
    float trail_arc_scale,
    uint32_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    bool oceanic = (is_oceanic[i] != 0);
    int pid = plate_id[i];

    float ax = plate_ax_x[pid], ay = plate_ax_y[pid], az = plate_ax_z[pid];
    float omega = plate_speed[pid];

    float local_elev = elevation[i];
    const int kTrailSteps = 6;

    for (int h = 0; h < num_hotspots; ++h) {
        bool deep = (hs_is_deep[h] != 0);
        if (!deep && !oceanic) continue;

        float hx0 = hs_x[h], hy0 = hs_y[h], hz0 = hs_z[h];
        float base_radius = hs_radius[h];
        float peak_meters = deep ? (hs_str[h] * 14.0f) : (hs_str[h] * 32.0f);
        if (!oceanic && deep) peak_meters = hs_str[h] * 11.5f;

        for (int s = 0; s < kTrailSteps; ++s) {
            float age_frac = static_cast<float>(s) / static_cast<float>(kTrailSteps - 1);
            float theta = omega * trail_arc_scale * age_frac;

            float cs = cosf(theta), sn = sinf(theta);
            float dot = ax * hx0 + ay * hy0 + az * hz0;
            float cx = ay * hz0 - az * hy0;
            float cy = az * hx0 - ax * hz0;
            float cz = ax * hy0 - ay * hx0;

            float hx = hx0 * cs + cx * sn + ax * dot * (1.0f - cs);
            float hy = hy0 * cs + cy * sn + ay * dot * (1.0f - cs);
            float hz = hz0 * cs + cz * sn + az * dot * (1.0f - cs);

            float dx = px - hx, dy = py - hy, dz = pz - hz;
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            float step_radius = base_radius * (1.0f - 0.35f * age_frac);
            if (dist < step_radius * 1.45f) {
                float noise = hash_fbm3d(px * 16.0f, py * 16.0f, pz * 16.0f, 3, seed + static_cast<uint32_t>(h * 31 + s));
                float dist_perturbed = dist * (1.0f - noise * 0.38f);
                if (dist_perturbed < step_radius) {
                    float t = 1.0f - dist_perturbed / step_radius;
                    float profile = t * t * (3.0f - 2.0f * t);
                    float age_decay = expf(-1.6f * age_frac);
                    local_elev += peak_meters * profile * age_decay * (0.85f + 0.30f * fabsf(noise));
                }
            }
        }
    }

    elevation[i] = local_elev;
}

// Detalle geomorfológico multifractal + cinturones montañosos intracontinentales
__global__ void tectonic_fbm_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float* elevation,
    const float* comp_stress,
    const uint8_t* is_oceanic,
    const int* dist_to_boundary,
    float hop_scale,
    uint32_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float s = fmaxf(0.0f, comp_stress[i]);
    float d_hops = static_cast<float>(dist_to_boundary[i]) * hop_scale;

    float spx = px * 2.0f, spy = py * 2.0f, spz = pz * 2.0f;

    float low_noise = hash_fbm3d(spx * 1.8f, spy * 1.8f, spz * 1.8f, 4, seed + 101u);
    if (low_noise > 0.30f) {
        low_noise = 0.30f + (low_noise - 0.30f) * 0.15f;
    }

    float ridge_hills  = hash_ridge3d(spx * 7.5f,  spy * 7.5f,  spz * 7.5f,  5, seed + 202u);
    float ridge_detail = hash_ridge3d(spx * 24.0f, spy * 24.0f, spz * 24.0f, 5, seed + 303u);
    float plains_micro = hash_fbm3d(spx * 18.0f, spy * 18.0f, spz * 18.0f, 3, seed + 404u);

    float stress_multiplier = fminf(1.2f, s * 0.70f);
    float current_elev = elevation[i];

    if (!is_oceanic[i]) {
        // Factor de interioridad: crece tierra adentro lejos de la costa para formar cordilleras intracratónicas
        float inland_factor = fminf(1.0f, d_hops * 0.35f);
        float interior_belt_mask = fmaxf(0.0f, hash_fbm3d(px * 3.2f, py * 3.2f, pz * 3.2f, 3, seed + 505u) - 0.15f);
        float interior_orogeny = inland_factor * interior_belt_mask * ridge_hills * ridge_hills * 2600.0f;

        current_elev += low_noise * 360.0f;
        current_elev += ridge_hills  * (220.0f + 1750.0f * stress_multiplier) + interior_orogeny;
        current_elev += ridge_detail * (95.0f  + 950.0f  * stress_multiplier + 450.0f * interior_belt_mask * inland_factor);
        current_elev += plains_micro * 45.0f;
        if (current_elev < 12.0f) current_elev = 12.0f; // Evita depresiones marinas falsas en corteza continental
    } else {
        current_elev += low_noise * 180.0f;
        current_elev += ridge_hills * (120.0f + 320.0f * stress_multiplier);
        current_elev += plains_micro * 60.0f;
    }

    elevation[i] = current_elev;
}

__global__ void tectonic_smooth_kernel(
    int num_cells,
    const float* elevation, float* next_elevation,
    const int* neighbors, const int* num_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    int nn = num_neighbors[i];
    float sum = 0.0f;
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        sum += elevation[nid];
    }
    float nbr_mean = (nn > 0) ? (sum / static_cast<float>(nn)) : elevation[i];
    next_elevation[i] = elevation[i] * 0.82f + nbr_mean * 0.18f;
}

// ============================================================================
// Métodos en Host (CPU)
// ============================================================================

TectonicSimulator::TectonicSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void TectonicSimulator::generatePlates(int numPlates, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    if (cells.empty()) return;

    const uint32_t u_seed = static_cast<uint32_t>(params.seed);
    std::mt19937 rng(params.seed);
    std::uniform_int_distribution<size_t> distNode(0, cells.size() - 1);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);

    plates.resize(numPlates);
    std::vector<bool> visited(cells.size(), false);

    Vector3 pangea_pole = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
    Vector3 secondary_pole = Math::normalize(Vector3(
        -pangea_pole.x + distReal(rng) * 0.5f,
        -pangea_pole.y + distReal(rng) * 0.5f,
        -pangea_pole.z + distReal(rng) * 0.5f
    ));

    float speed_scale = params.tectonic_speed();
    int numMajor = std::max(1, static_cast<int>(numPlates * 0.72f));

    for (int i = 0; i < numMajor; ++i) {
        bool is_oceanic_plate = distProb(rng) > params.crust_fraction;

        size_t seedId;
        if (!is_oceanic_plate) {
            float cluster_strength = params.primary_clustering * 0.65f;
            float secondary_strength = params.secondary_clustering * 0.65f;

            size_t best = distNode(rng);
            float best_score = 1e10f;
            int attempts = static_cast<int>(50 * cluster_strength) + 2;

            for (int a = 0; a < attempts; ++a) {
                size_t candidate = distNode(rng);
                if (visited[candidate]) continue;

                float dist_primary = static_cast<float>(Math::distance(cells[candidate].position, pangea_pole));
                float dist_secondary = static_cast<float>(Math::distance(cells[candidate].position, secondary_pole));

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
        plates[i].rotation_axis = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));

        if (plates[i].is_oceanic) {
            plates[i].angular_speed = (0.4f + 0.6f * distProb(rng)) * speed_scale * 0.22f;
        } else {
            plates[i].angular_speed = (0.3f + 0.7f * distProb(rng)) * speed_scale * 0.12f;
        }
    }

    auto assign_warped_voronoi = [&](int nseeds) {
        const float wa = 0.06f + 0.30f * params.crust_warping;
        for (auto& cell : cells) {
            float px = static_cast<float>(cell.position.x);
            float py = static_cast<float>(cell.position.y);
            float pz = static_cast<float>(cell.position.z);
            float wx = px + wa * cpu_hash_fbm3d(px * 2.4f + 11.3f, py * 2.4f,         pz * 2.4f,         4, u_seed + 1u);
            float wy = py + wa * cpu_hash_fbm3d(px * 2.4f,         py * 2.4f + 24.7f, pz * 2.4f,         4, u_seed + 2u);
            float wz = pz + wa * cpu_hash_fbm3d(px * 2.4f,         py * 2.4f,         pz * 2.4f + 39.1f, 4, u_seed + 3u);
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

    assign_warped_voronoi(numMajor);

    std::vector<size_t> boundary_cells;
    for (size_t i = 0; i < cells.size(); ++i) {
        int pid = cells[i].plate_id;
        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].plate_id != pid) {
                boundary_cells.push_back(i);
                break;
            }
        }
    }

    if (!boundary_cells.empty() && numMajor < numPlates) {
        std::uniform_int_distribution<size_t> distBoundary(0, boundary_cells.size() - 1);
        for (int i = numMajor; i < numPlates; ++i) {
            size_t seedId = boundary_cells[distBoundary(rng)];
            plates[i].is_oceanic = distProb(rng) > params.crust_fraction;
            plates[i].center = cells[seedId].position;
            plates[i].rotation_axis = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
            if (plates[i].is_oceanic) {
                plates[i].angular_speed = (0.5f + 0.5f * distProb(rng)) * speed_scale * 0.28f;
            } else {
                plates[i].angular_speed = (0.4f + 0.6f * distProb(rng)) * speed_scale * 0.16f;
            }
        }
        assign_warped_voronoi(numPlates);
    }

    std::uniform_real_distribution<float> distContVar(-40.0f, 80.0f);
    float warp = params.crust_warping;

    for (auto& cell : cells) {
        int pid = cell.plate_id;
        float px = static_cast<float>(cell.position.x);
        float py = static_cast<float>(cell.position.y);
        float pz = static_cast<float>(cell.position.z);

        // Reducir el margen oceánico forzado para permitir que las placas continentales vecinas
        // choquen tierra adentro (formando cordilleras interiores tipo Himalaya/Urales).
        float margin_noise = cpu_hash_fbm3d(px * 3.2f, py * 3.2f, pz * 3.2f, 3, u_seed + 77u);
        float dist_from_center = static_cast<float>(Math::distance(cell.position, plates[pid].center));
        bool mixed_oceanic_margin = (!plates[pid].is_oceanic) &&
                                    (dist_from_center > 0.78f + 0.25f * margin_noise) &&
                                    (params.crust_fraction < 0.50f);

        cell.is_oceanic = plates[pid].is_oceanic || mixed_oceanic_margin;

        if (cell.is_oceanic) {
            float age_noise = 0.5f + 0.5f * cpu_hash_fbm3d(px * 4.0f, py * 4.0f, pz * 4.0f, 3, u_seed + 88u);
            cell.crustal_age = std::max(2.0f, std::min(160.0f, (dist_from_center * 95.0f + age_noise * 45.0f)));
            cell.crustal_thickness = 7.0f;
            cell.elevation = -2500.0f - params.thermal_subsidence_rate * std::sqrt(cell.crustal_age);
            cell.bedrock = RockType::BASALT;
        } else {
            float craton_noise = std::abs(cpu_hash_fbm3d(px * 6.0f, py * 6.0f, pz * 6.0f, 4, u_seed + 99u));
            float warp_noise = std::abs(cpu_hash_noise3d(px * 3.0f, py * 3.0f, pz * 3.0f, u_seed + 111u)) * warp;
            float shield = std::pow(std::min(1.0f, craton_noise + 0.35f * warp_noise), 2.1f);

            cell.elevation = 90.0f + shield * 550.0f + distContVar(rng);
            cell.bedrock = RockType::GRANITE;
            cell.crustal_thickness = 35.0f + (cell.elevation - 90.0f) * 0.008f;
            cell.crustal_age = 1500.0f + shield * 1500.0f;
        }

        Vector3 w(
            plates[pid].rotation_axis.x * plates[pid].angular_speed,
            plates[pid].rotation_axis.y * plates[pid].angular_speed,
            plates[pid].rotation_axis.z * plates[pid].angular_speed
        );
        cell.plate_velocity = Math::crossProduct(w, cell.position);
    }
}

void TectonicSimulator::applyTerrainFeatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    if (cells.empty()) return;

    const uint32_t u_seed = static_cast<uint32_t>(params.seed);
    std::mt19937 rng(params.seed + 7777);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distProb(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> distCell(0, cells.size() - 1);

    // Selecciona una semilla preferentemente en el interior de un continente (lejos de celdas oceánicas)
    auto pick_inland_center = [&]() -> Vector3 {
        Vector3 best = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        for (int attempt = 0; attempt < 40; ++attempt) {
            size_t idx = distCell(rng);
            if (cells[idx].is_oceanic) continue;
            bool coastal = false;
            for (size_t nid : cells[idx].neighbors) {
                if (cells[nid].is_oceanic) { coastal = true; break; }
            }
            if (!coastal) return cells[idx].position;
            best = cells[idx].position;
        }
        return best;
    };

    auto add_dome = [&](float ang_radius, float peak, float rim, bool land_only, uint32_t tag) {
        Vector3 c = land_only ? pick_inland_center()
                              : Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
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

            float wob = cpu_hash_fbm3d(px * 4.0f + phase, py * 4.0f, pz * 4.0f, 3, u_seed + tag);
            float r = ang_radius * (1.0f + 0.4f * wob);
            if (chord >= r) continue;

            float t = chord / r;
            float falloff = (1.0f - t) * (1.0f - t);
            float uplift = peak * falloff + rim * (1.0f - falloff);
            uplift *= 0.75f + 0.25f * std::abs(cpu_hash_noise3d(px * 13.0f, py * 13.0f, pz * 13.0f, u_seed + tag + 1u));

            cell.elevation += uplift;
            if (!cell.is_oceanic) cell.crustal_thickness += uplift / 120.0f;
        }
    };

    // Cordilleras intracontinentales (ej. Rocosas, Urales, Apalaches, Tian Shan):
    // Se siembran en el interior continental y se esculpen con ridge noise para lucir como cadenas reales.
    auto add_ridge = [&](float arc_len, float half_width, float peak, float rim, bool land_only, uint32_t tag) {
        Vector3 a = land_only ? pick_inland_center()
                              : Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        Vector3 dir = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        double adot = Math::dotProduct(a, dir);
        Vector3 tang = Math::normalize(Vector3(dir.x - adot * a.x, dir.y - adot * a.y, dir.z - adot * a.z));
        float cs = std::cos(arc_len), sn = std::sin(arc_len);
        Vector3 b = Math::normalize(Vector3(a.x * cs + tang.x * sn, a.y * cs + tang.y * sn, a.z * cs + tang.z * sn));

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

            float apx = px - static_cast<float>(a.x);
            float apy = py - static_cast<float>(a.y);
            float apz = pz - static_cast<float>(a.z);
            float u = std::max(0.0f, std::min(1.0f, (apx * abx + apy * aby + apz * abz) / ab2));

            // Curvar el eje de la cordillera con ruido para que no sea un segmento recto
            float curve_x = cpu_hash_fbm3d(px * 4.5f + phase, py * 4.5f, pz * 4.5f, 3, u_seed + tag + 5u) * 0.045f;
            float curve_y = cpu_hash_fbm3d(px * 4.5f, py * 4.5f + phase, pz * 4.5f, 3, u_seed + tag + 6u) * 0.045f;
            float dx = apx - abx * u + curve_x;
            float dy = apy - aby * u + curve_y;
            float dz = apz - abz * u;
            float d = std::sqrt(dx * dx + dy * dy + dz * dz);

            float wob = cpu_hash_fbm3d(px * 6.0f + phase, py * 6.0f, pz * 6.0f, 3, u_seed + tag);
            float w = half_width * (1.0f + 0.45f * wob);
            if (d >= w) continue;

            // Atenuar en los dos extremos del arco y si se acerca a la costa
            float end_taper = std::sin(u * static_cast<float>(Math::PI));
            float t = d / w;
            float falloff = (1.0f - t * t) * (1.0f - t * t) * end_taper;
            float ridge_crest = cpu_hash_ridge3d(px * 22.0f, py * 22.0f, pz * 22.0f, 4, u_seed + tag + 1u);
            float uplift = (peak * (0.45f + 0.75f * ridge_crest) * falloff + rim * (1.0f - t));

            cell.elevation += uplift;
            if (!cell.is_oceanic) cell.crustal_thickness += uplift / 120.0f;
        }
    };

    for (int i = 0; i < params.superswell_freq; ++i)   add_dome(0.48f,  620.0f,   0.0f, false, 1000u + i * 2u);
    for (int i = 0; i < params.old_mountain_freq; ++i) add_ridge(0.55f, 0.052f, 2600.0f, 180.0f, true, 2000u + i * 2u);
    for (int i = 0; i < params.old_hill_freq; ++i)     add_ridge(0.38f, 0.045f, 1100.0f,  90.0f, true, 3000u + i * 2u);
    for (int i = 0; i < params.small_uplift_freq; ++i) add_dome(0.080f, 420.0f,   50.0f, true, 4000u + i * 2u);
    for (int i = 0; i < params.upland_freq; ++i)       add_dome(0.34f,  320.0f,   80.0f, true, 5000u + i * 2u);
}

void TectonicSimulator::generateHotspots(const SimulationParameters& params) {
    std::mt19937 rng(params.seed + 4242);
    std::uniform_real_distribution<float> distReal(-1.0f, 1.0f);

    hotspots.clear();

    for (int i = 0; i < params.deep_plume_freq; ++i) {
        Hotspot h;
        h.position = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        h.strength = 140.0f + distReal(rng) * 45.0f;
        h.radius = 0.13f;
        h.is_deep = true;
        hotspots.push_back(h);
    }

    for (int i = 0; i < params.shallow_plume_freq; ++i) {
        Hotspot h;
        h.position = Math::normalize(Vector3(distReal(rng), distReal(rng), distReal(rng)));
        h.strength = 95.0f + distReal(rng) * 35.0f;
        h.radius = 0.065f;
        h.is_deep = false;
        hotspots.push_back(h);
    }
}

void TectonicSimulator::simulate(int iterations, const SimulationParameters& params) {
    generateHotspots(params);

    auto& cells = planet.getCells();
    int num_cells = static_cast<int>(cells.size());
    if (num_cells == 0) return;

    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_elev(num_cells), h_vel_x(num_cells), h_vel_y(num_cells), h_vel_z(num_cells);
    std::vector<int> h_plate_id(num_cells);
    std::vector<uint8_t> h_is_oceanic(num_cells);
    std::vector<float> h_crustal_age(num_cells), h_crustal_thickness(num_cells);

    for (int i = 0; i < num_cells; ++i) {
        h_pos_x[i] = static_cast<float>(cells[i].position.x);
        h_pos_y[i] = static_cast<float>(cells[i].position.y);
        h_pos_z[i] = static_cast<float>(cells[i].position.z);
        h_elev[i] = cells[i].elevation;
        h_vel_x[i] = static_cast<float>(cells[i].plate_velocity.x);
        h_vel_y[i] = static_cast<float>(cells[i].plate_velocity.y);
        h_vel_z[i] = static_cast<float>(cells[i].plate_velocity.z);
        h_plate_id[i] = cells[i].plate_id;
        h_is_oceanic[i] = cells[i].is_oceanic ? 1 : 0;
        h_crustal_age[i] = cells[i].crustal_age;
        h_crustal_thickness[i] = cells[i].crustal_thickness;
    }

    const auto& h_neighbors = planet.getRowMajorNeighbors();
    const auto& h_num_neighbors = planet.getNeighborCounts();

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_elev, *d_next_elev;
    float *d_vel_x, *d_vel_y, *d_vel_z;
    float *d_age, *d_next_age, *d_thick, *d_next_thick;
    int *d_plate_id, *d_neighbors, *d_num_neighbors;
    uint8_t *d_is_oceanic, *d_next_is_oceanic;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vel_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_age, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_age, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_thick, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_thick, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_id, num_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_is_oceanic, num_cells * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_next_is_oceanic, num_cells * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

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

    int numPlates = static_cast<int>(plates.size());
    float *d_plate_axes_x, *d_plate_axes_y, *d_plate_axes_z, *d_plate_speeds;
    float *d_plate_centers_x, *d_plate_centers_y, *d_plate_centers_z;
    uint8_t *d_plate_is_oceanic;

    CHECK_CUDA(cudaMalloc(&d_plate_axes_x, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_axes_y, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_axes_z, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_speeds, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_x, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_y, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_centers_z, numPlates * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_plate_is_oceanic, numPlates * sizeof(uint8_t)));

    std::vector<float> h_plate_axes_x(numPlates), h_plate_axes_y(numPlates), h_plate_axes_z(numPlates);
    std::vector<float> h_plate_speeds(numPlates);
    std::vector<float> h_plate_centers_x(numPlates), h_plate_centers_y(numPlates), h_plate_centers_z(numPlates);
    std::vector<uint8_t> h_plate_oceanic(numPlates);

    for (int p = 0; p < numPlates; ++p) {
        h_plate_axes_x[p] = static_cast<float>(plates[p].rotation_axis.x);
        h_plate_axes_y[p] = static_cast<float>(plates[p].rotation_axis.y);
        h_plate_axes_z[p] = static_cast<float>(plates[p].rotation_axis.z);
        h_plate_speeds[p] = plates[p].angular_speed;
        h_plate_centers_x[p] = static_cast<float>(plates[p].center.x);
        h_plate_centers_y[p] = static_cast<float>(plates[p].center.y);
        h_plate_centers_z[p] = static_cast<float>(plates[p].center.z);
        h_plate_oceanic[p] = plates[p].is_oceanic ? 1 : 0;
    }

    CHECK_CUDA(cudaMemcpy(d_plate_axes_x, h_plate_axes_x.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_axes_y, h_plate_axes_y.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_axes_z, h_plate_axes_z.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_speeds, h_plate_speeds.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_x, h_plate_centers_x.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_y, h_plate_centers_y.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_centers_z, h_plate_centers_z.data(), numPlates * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_plate_is_oceanic, h_plate_oceanic.data(), numPlates * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float *d_comp_stress, *d_next_comp_stress;
    float *d_div_stress, *d_next_div_stress;
    float *d_shear_stress, *d_next_shear_stress;
    uint8_t *d_is_collision, *d_next_is_collision;
    int *d_dist_boundary, *d_next_dist_boundary;

    CHECK_CUDA(cudaMalloc(&d_comp_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_comp_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_div_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_div_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_shear_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_shear_stress, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_is_collision, num_cells * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_next_is_collision, num_cells * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_dist_boundary, num_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_next_dist_boundary, num_cells * sizeof(int)));

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    float ref_speed = 0.05f * 0.18f;
    float speed_norm_inv = 1.0f / std::max(1e-5f, ref_speed);

    int res_mult = 1 << std::max(0, params.subdivision_level - 5);
    float hop_scale = 1.0f / static_cast<float>(res_mult);
    float step_decay = std::pow(0.86f, hop_scale);

    if (params.use_bisector_distance) {
        tectonic_bisector_stress_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_plate_id, d_is_oceanic, d_age,
            d_comp_stress, d_div_stress, d_shear_stress,
            d_is_collision, d_dist_boundary,
            numPlates,
            d_plate_centers_x, d_plate_centers_y, d_plate_centers_z,
            d_plate_axes_x, d_plate_axes_y, d_plate_axes_z,
            d_plate_speeds, d_plate_is_oceanic,
            speed_norm_inv, res_mult
        );
    } else {
        tectonic_boundary_kinematics_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_plate_id, d_vel_x, d_vel_y, d_vel_z,
            d_is_oceanic, d_age,
            d_comp_stress, d_div_stress, d_shear_stress,
            d_is_collision, d_dist_boundary,
            d_neighbors, d_num_neighbors,
            speed_norm_inv
        );

        int propagation_steps = 12 * res_mult;
        for (int s = 0; s < propagation_steps; ++s) {
            tectonic_propagate_stress_kernel<<<numBlocks, blockSize>>>(
                num_cells, d_plate_id,
                d_comp_stress, d_next_comp_stress,
                d_div_stress, d_next_div_stress,
                d_shear_stress, d_next_shear_stress,
                d_is_collision, d_next_is_collision,
                d_dist_boundary, d_next_dist_boundary,
                d_neighbors, d_num_neighbors,
                step_decay
            );
            std::swap(d_comp_stress, d_next_comp_stress);
            std::swap(d_div_stress, d_next_div_stress);
            std::swap(d_shear_stress, d_next_shear_stress);
            std::swap(d_is_collision, d_next_is_collision);
            std::swap(d_dist_boundary, d_next_dist_boundary);
        }
    }

    int macro_epochs = std::max(4, std::min(32, iterations / 4));
    float active_window_Myr = std::min(params.planet_age_Myr, 180.0f);
    float dt_Myr = active_window_Myr / static_cast<float>(macro_epochs);
    float step_rad_scale = 0.35f * (dt_Myr / 10.0f);

    for (int ep = 0; ep < macro_epochs; ++ep) {
        tectonic_advect_and_age_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_plate_id, d_vel_x, d_vel_y, d_vel_z,
            d_is_oceanic, d_next_is_oceanic,
            d_age, d_next_age,
            d_thick, d_next_thick,
            d_div_stress, d_dist_boundary,
            d_neighbors, d_num_neighbors,
            dt_Myr, step_rad_scale, hop_scale
        );
        std::swap(d_is_oceanic, d_next_is_oceanic);
        std::swap(d_age, d_next_age);
        std::swap(d_thick, d_next_thick);
    }

    tectonic_synthesize_orogeny_kernel<<<numBlocks, blockSize>>>(
        num_cells, d_pos_x, d_pos_y, d_pos_z,
        d_elev, d_thick, d_is_oceanic, d_age,
        d_comp_stress, d_div_stress, d_shear_stress,
        d_is_collision, d_dist_boundary,
        d_neighbors, d_num_neighbors,
        params.orogenesis_factor,
        params.thermal_subsidence_rate,
        u_seed, hop_scale
    );

    int num_hotspots = static_cast<int>(hotspots.size());
    float *d_hs_x = nullptr, *d_hs_y = nullptr, *d_hs_z = nullptr;
    float *d_hs_str = nullptr, *d_hs_radius = nullptr;
    uint8_t *d_hs_deep = nullptr;

    if (num_hotspots > 0) {
        CHECK_CUDA(cudaMalloc(&d_hs_x, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_y, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_z, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_str, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_radius, num_hotspots * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hs_deep, num_hotspots * sizeof(uint8_t)));

        std::vector<float> h_hs_x(num_hotspots), h_hs_y(num_hotspots), h_hs_z(num_hotspots);
        std::vector<float> h_hs_str(num_hotspots), h_hs_rad(num_hotspots);
        std::vector<uint8_t> h_hs_deep(num_hotspots);
        for (int h = 0; h < num_hotspots; ++h) {
            h_hs_x[h] = static_cast<float>(hotspots[h].position.x);
            h_hs_y[h] = static_cast<float>(hotspots[h].position.y);
            h_hs_z[h] = static_cast<float>(hotspots[h].position.z);
            h_hs_str[h] = hotspots[h].strength;
            h_hs_rad[h] = hotspots[h].radius;
            h_hs_deep[h] = hotspots[h].is_deep ? 1 : 0;
        }
        CHECK_CUDA(cudaMemcpy(d_hs_x, h_hs_x.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_y, h_hs_y.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_z, h_hs_z.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_str, h_hs_str.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_radius, h_hs_rad.data(), num_hotspots * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_hs_deep, h_hs_deep.data(), num_hotspots * sizeof(uint8_t), cudaMemcpyHostToDevice));

        float trail_arc_scale = std::min(25.0f, std::max(4.0f, params.planet_age_Myr * 0.04f));
        tectonic_hotspot_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_elev, d_is_oceanic, d_plate_id,
            d_plate_axes_x, d_plate_axes_y, d_plate_axes_z, d_plate_speeds,
            num_hotspots, d_hs_x, d_hs_y, d_hs_z, d_hs_str, d_hs_radius, d_hs_deep,
            trail_arc_scale, u_seed
        );
    }

    tectonic_fbm_kernel<<<numBlocks, blockSize>>>(
        num_cells, d_pos_x, d_pos_y, d_pos_z,
        d_elev, d_comp_stress, d_is_oceanic,
        d_dist_boundary, hop_scale, u_seed
    );

    int smooth_passes = 3 * std::max(1, res_mult / 2);
    for (int s = 0; s < smooth_passes; ++s) {
        tectonic_smooth_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_elev, d_next_elev, d_neighbors, d_num_neighbors
        );
        std::swap(d_elev, d_next_elev);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_elev.data(), d_elev, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_crustal_thickness.data(), d_thick, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_is_oceanic.data(), d_is_oceanic, num_cells * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_crustal_age.data(), d_age, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cells; ++i) {
        float e = h_elev[i];
        if (!std::isfinite(e)) e = -3500.0f;
        cells[i].elevation = e;
        cells[i].crustal_thickness = h_crustal_thickness[i];
        cells[i].is_oceanic = (h_is_oceanic[i] == 1);
        cells[i].crustal_age = h_crustal_age[i];
        if (cells[i].elevation > 3200.0f || cells[i].crustal_thickness > 48.0f) {
            cells[i].bedrock = RockType::METAMORPHIC;
        }
    }

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
                int c = fq.front();
                fq.pop();
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
            for (int c = 1; c < static_cast<int>(compSize.size()); ++c) {
                if (compSize[c] > compSize[mainComp]) mainComp = c;
            }
            const float inland_floor = sea + 8.0f;
            for (int i = 0; i < num_cells; ++i) {
                if (comp[i] != -1 && comp[i] != mainComp && cells[i].elevation < inland_floor) {
                    cells[i].elevation = inland_floor;
                    cells[i].is_oceanic = false;
                }
            }
        }
    }

    applyTerrainFeatures(params);

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_elev); cudaFree(d_next_elev);
    cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
    cudaFree(d_age); cudaFree(d_next_age);
    cudaFree(d_thick); cudaFree(d_next_thick);
    cudaFree(d_plate_id);
    cudaFree(d_is_oceanic); cudaFree(d_next_is_oceanic);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
    cudaFree(d_plate_axes_x); cudaFree(d_plate_axes_y); cudaFree(d_plate_axes_z);
    cudaFree(d_plate_speeds);
    cudaFree(d_plate_centers_x); cudaFree(d_plate_centers_y); cudaFree(d_plate_centers_z);
    cudaFree(d_plate_is_oceanic);
    cudaFree(d_comp_stress); cudaFree(d_next_comp_stress);
    cudaFree(d_div_stress); cudaFree(d_next_div_stress);
    cudaFree(d_shear_stress); cudaFree(d_next_shear_stress);
    cudaFree(d_is_collision); cudaFree(d_next_is_collision);
    cudaFree(d_dist_boundary); cudaFree(d_next_dist_boundary);

    if (num_hotspots > 0) {
        cudaFree(d_hs_x); cudaFree(d_hs_y); cudaFree(d_hs_z);
        cudaFree(d_hs_str); cudaFree(d_hs_radius); cudaFree(d_hs_deep);
    }
}

} // namespace Ravis