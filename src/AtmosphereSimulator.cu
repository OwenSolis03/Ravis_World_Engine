#define _USE_MATH_DEFINES
#include "../include/AtmosphereSimulator.h"
#include "../include/MathUtils.h"
#include "../include/Noise.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
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
} while (0)

// ============================================================================
// Kernels CUDA para Dinámica Atmosférica (SWE + Transporte de Humedad)
// ============================================================================

__global__ void wind_update_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* lat,
    const float* elevation,
    float* wind_u, float* wind_v, float* wind_w,
    const float* pressure,
    const int* neighbors, const int* num_neighbors,
    float sea_level, float dt, float grad_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float phi = lat[i];
    float p = pressure[i];
    float my_elev = fmaxf(0.0f, elevation[i] - sea_level);

    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
    int nn = num_neighbors[i];
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        float ux = dx / dist, uy = dy / dist, uz = dz / dist;

        float np = pressure[nid];
        float nbr_elev = fmaxf(0.0f, elevation[nid] - sea_level);
        float topo_barrier = (nbr_elev - my_elev) * 0.000022f;

        float diff = ((np - p) + topo_barrier) * grad_scale;
        grad_x += ux * diff;
        grad_y += uy * diff;
        grad_z += uz * diff;
    }
    if (nn > 0) {
        float inv_n = 1.0f / static_cast<float>(nn);
        grad_x *= inv_n; grad_y *= inv_n; grad_z *= inv_n;
    }

    float xy = sqrtf(px * px + py * py);
    float east_x = (xy > 1e-5f) ? (-py / xy) : 1.0f;
    float east_y = (xy > 1e-5f) ? ( px / xy) : 0.0f;
    float east_z = 0.0f;

    float north_x = py * east_z - pz * east_y;
    float north_y = pz * east_x - px * east_z;
    float north_z = px * east_y - py * east_x;

    // Vientos planetarios con desviación meridional para romper la horizontalidad pura
    float u_zonal_eq = -cosf(3.0f * phi) * 0.85f;
    float v_merid_eq = -sinf(3.0f * phi) * 0.32f * (phi >= 0.0f ? 1.0f : -1.0f);
    float eq_wx = u_zonal_eq * east_x + v_merid_eq * north_x;
    float eq_wy = u_zonal_eq * east_y + v_merid_eq * north_y;
    float eq_wz = u_zonal_eq * east_z + v_merid_eq * north_z;

    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];
    float f = 2.1f * sinf(phi);

    float drag = (elevation[i] > sea_level) ? (0.032f + fminf(0.045f, my_elev * 0.000010f)) : 0.014f;

    float cor_x = f * (py * ww - pz * wv);
    float cor_y = f * (pz * wu - px * ww);
    float cor_z = f * (px * wv - py * wu);

    // Mayor peso al gradiente de presión térmico continente-mar (monzones y giros subtropicales)
    wu = wu * (1.0f - drag) - grad_x * dt * 1.35f + cor_x * dt + (eq_wx - wu) * 0.08f;
    wv = wv * (1.0f - drag) - grad_y * dt * 1.35f + cor_y * dt + (eq_wy - wv) * 0.08f;
    ww = ww * (1.0f - drag) - grad_z * dt * 1.35f + cor_z * dt + (eq_wz - ww) * 0.08f;

    float dot = wu * px + wv * py + ww * pz;
    wu -= dot * px;
    wv -= dot * py;
    ww -= dot * pz;

    float speed = sqrtf(wu * wu + wv * wv + ww * ww);
    if (speed > 1.35f) {
        float s = 1.35f / speed;
        wu *= s; wv *= s; ww *= s;
    }

    wind_u[i] = wu;
    wind_v[i] = wv;
    wind_w[i] = ww;
}

__global__ void pressure_update_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* wind_u, const float* wind_v, const float* wind_w,
    const float* pressure, const float* eq_pressure,
    float* next_pressure, float* out_divergence,
    const int* neighbors, const int* num_neighbors,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];

    float div = 0.0f;
    int nn = num_neighbors[i];
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        float vx = dx / dist, vy = dy / dist, vz = dz / dist;

        float out_w = wu * vx + wv * vy + ww * vz;
        float in_w  = -(wind_u[nid] * vx + wind_v[nid] * vy + wind_w[nid] * vz);
        div += (out_w - in_w);
    }
    if (nn > 0) div /= static_cast<float>(nn);

    out_divergence[i] = div;
    float p_new = pressure[i] - div * dt * 0.14f + (eq_pressure[i] - pressure[i]) * 0.12f;
    next_pressure[i] = fmaxf(0.0f, fminf(2.0f, p_new));
}

// Kernel de advección y precipitación sin bandas rígidas:
// - Utiliza ondas planetarias de Rossby y ruido regional 3D para curvar las zonas climáticas (+/- 18°).
// - Genera asimetría Este-Oeste (costas orientales subtropicales húmedas por vientos alisios marítimos,
//   costas occidentales e interiores continentales áridos).
// - Suprime la humedad en sombras orográficas y en el corazón de grandes supercontinentes.
__global__ void moisture_advect_precip_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* lat,
    const float* elevation,
    const float* temperature,
    const float* sst_anomaly,
    const float* wind_u, const float* wind_v, const float* wind_w,
    const float* wind_div,
    const float* moisture_in, float* moisture_out,
    float* precip_accum,
    const int* neighbors, const int* num_neighbors,
    float sea_level, float cfl_advect, float step_scale, uint32_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float phi = lat[i];
    float lon = atan2f(py, px);
    float lat_deg = fabsf(phi) * 57.2957795f;
    float elev = elevation[i];
    float temp = temperature[i];

    float q_cap = 0.42f + 0.58f * fmaxf(0.0f, fminf(1.0f, temp));
    float elev_above = fmaxf(0.0f, elev - sea_level);
    // Las grandes cordilleras exprimen el vapor por enfriamiento adiabático, secando el aire que pasa al otro lado
    q_cap *= expf(-elev_above * 0.00024f);
    q_cap = fmaxf(0.12f, q_cap);

    int nn = num_neighbors[i];
    float incoming_flux = 0.0f;
    float outgoing_weight_sum = 0.0f;
    float mean_upslope_m = 0.0f;
    float downslope_m = 0.0f;

    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        dx /= dist; dy /= dist; dz /= dist;

        float dot_out = wu * dx + wv * dy + ww * dz;
        if (dot_out > 0.0f) {
            outgoing_weight_sum += dot_out;
            float dh = elevation[nid] - elev;
            if (dh > 0.0f) {
                mean_upslope_m += dh * dot_out;
            } else {
                downslope_m += (-dh) * dot_out;
            }
        }

        float nwu = wind_u[nid], nwv = wind_v[nid], nww = wind_w[nid];
        float dot_from_nbr = -(nwu * dx + nwv * dy + nww * dz);
        if (dot_from_nbr > 0.0f) {
            float nbr_out_sum = 0.0f;
            int n_nn = num_neighbors[nid];
            float npx = pos_x[nid], npy = pos_y[nid], npz = pos_z[nid];
            for (int k = 0; k < n_nn; ++k) {
                int knid = neighbors[nid * 6 + k];
                float kdx = pos_x[knid] - npx;
                float kdy = pos_y[knid] - npy;
                float kdz = pos_z[knid] - npz;
                float kdist = sqrtf(kdx * kdx + kdy * kdy + kdz * kdz + 1e-8f);
                float kdot = (nwu * kdx + nwv * kdy + nww * kdz) / kdist;
                if (kdot > 0.0f) nbr_out_sum += kdot;
            }
            if (nbr_out_sum > 1e-6f) {
                float nbr_speed = fminf(0.96f, (0.45f + 0.55f * sqrtf(nwu * nwu + nwv * nwv + nww * nww)) * cfl_advect);
                float share = (dot_from_nbr / nbr_out_sum) * nbr_speed;
                incoming_flux += moisture_in[nid] * share;
            }
        }
    }

    if (outgoing_weight_sum > 1e-6f) {
        mean_upslope_m /= outgoing_weight_sum;
        downslope_m /= outgoing_weight_sum;
    }

    float my_speed = fminf(0.96f, (0.45f + 0.55f * sqrtf(wu * wu + wv * wv + ww * ww)) * cfl_advect);
    float retained = (outgoing_weight_sum > 1e-6f) ? (moisture_in[i] * (1.0f - my_speed)) : moisture_in[i];

    float nbr_mean = 0.0f;
    for (int n = 0; n < nn; ++n) {
        nbr_mean += moisture_in[neighbors[i * 6 + n]];
    }
    if (nn > 0) nbr_mean /= static_cast<float>(nn);

    float q = 0.75f * (retained + incoming_flux) + 0.25f * nbr_mean;

    // Distorsión planetaria fuerte: Ondas estacionarias de Rossby (números de onda 2 y 3) +
    // ruido 3D multiescala para romper por completo el aspecto de franjas horizontales.
    float rossby_wave = 9.5f * sinf(2.0f * lon + 1.3f) * sinf(2.0f * phi)
                      + 6.5f * cosf(3.0f * lon - 0.8f) * cosf(phi);
    float regional_noise = rw_fbm3d(px * 1.9f, py * 1.9f, pz * 1.9f, 4, seed + 701u) * 16.0f;
    float micro_climate  = rw_fbm3d(px * 5.5f, py * 5.5f, pz * 5.5f, 3, seed + 887u);

    float eff_lat = fmaxf(0.0f, fminf(90.0f, lat_deg + rossby_wave + regional_noise));

    // Parches climáticos orgánicos en vez de líneas duras:
    float itcz_band    = expf(-0.5f * (eff_lat / 15.0f) * (eff_lat / 15.0f));
    float midlat_storm = 0.82f * expf(-0.5f * ((eff_lat - 50.0f) / 16.0f) * ((eff_lat - 50.0f) / 16.0f));

    // La subsidencia subtropical (altas presiones) se rompe con variaciones longitudinales (celdas de alta presión
    // aisladas estilo anticiclón de las Azores/Pacífico en vez de un anillo continuo alrededor del globo).
    float gyre_cell_mod = fmaxf(0.15f, 0.65f + 0.55f * rw_fbm3d(px * 1.4f + 5.0f, py * 1.4f, pz * 1.4f, 3, seed + 911u));
    float subtrop_high  = expf(-0.5f * ((eff_lat - 26.5f) / 11.0f) * ((eff_lat - 26.5f) / 11.0f)) * gyre_cell_mod;

    float div = wind_div[i];
    float dynamic_conv = fmaxf(0.0f, -div);
    float dynamic_subs = fmaxf(0.0f,  div);

    // 1. Fuentes y sumideros de humedad (Bomba Biótica y Evaporación Marina):
    if (elev <= sea_level) {
        float sst_boost = 1.0f + sst_anomaly[i] * 2.2f;
        float evap = q_cap * (0.34f + 0.20f * my_speed) * fmaxf(0.35f, sst_boost) * step_scale;
        q = fminf(q_cap * 1.08f, q + evap);
    } else {
        // Bomba biótica continental: donde entra humedad marítima, la vegetación recicla
        // el vapor viento abajo permitiendo que las selvas y bosques penetren miles de km.
        float wet_canopy = fminf(1.0f, q / (q_cap * 0.45f));
        float recycle_rate = (0.042f + 0.065f * wet_canopy * (0.45f + 0.55f * (itcz_band + midlat_storm)))
                           * (1.0f - 0.45f * subtrop_high);
        // Sombra orográfica Foehn al descender cordilleras y secado en centros de divergencia anticiclónica
        float foehn_drying = fminf(0.065f, (downslope_m / 550.0f) * 0.050f);
        float subs_drying  = 0.014f * subtrop_high * dynamic_subs * (1.0f - 0.4f * micro_climate);
        q = fmaxf(0.0f, fminf(q_cap * 1.05f, q + (q * recycle_rate - foehn_drying - q * subs_drying) * step_scale));
    }

    // 2. Precipitación física:
    float excess_sat = fmaxf(0.0f, q - q_cap) * 0.40f * step_scale;

    float orographic_factor = 0.0f;
    if (elev > sea_level && mean_upslope_m > 0.0f) {
        orographic_factor = fminf(0.32f, (mean_upslope_m / 950.0f) * 0.25f * (0.5f + 0.5f * my_speed));
    }
    float lee_shadow = expf(-downslope_m / 240.0f);

    float humidity_non_linear = fminf(1.35f, (q / q_cap) * 1.25f);
    float synoptic_lift = (0.32f * itcz_band + 0.34f * midlat_storm + 0.26f * dynamic_conv + 0.08f * micro_climate)
                        * (1.0f - 0.52f * subtrop_high);
    synoptic_lift = fmaxf(0.03f, synoptic_lift);

    float rain_orographic = q * orographic_factor * step_scale;
    float rain_synoptic   = q * humidity_non_linear * synoptic_lift * 0.22f * lee_shadow * step_scale;
    float rain_base       = q * humidity_non_linear * 0.038f * (1.0f - 0.60f * subtrop_high) * lee_shadow * step_scale;

    float total_rain = excess_sat + rain_orographic + rain_synoptic + rain_base;
    total_rain = fminf(q * 0.38f * step_scale, total_rain);

    // Pérdida suave en columna (0.24 en tierra) para alcanzar ~3000 km de penetración continental
    float depletion = total_rain * (elev > sea_level ? 0.24f : 0.18f);
    q = fmaxf(0.0f, fminf(1.25f, q - depletion));

    moisture_out[i] = q;
    precip_accum[i] += total_rain;
}

// ============================================================================
// Métodos en Host (CPU)
// ============================================================================

AtmosphereSimulator::AtmosphereSimulator(GoldbergPolyhedron& planet) : planet(planet) {}

void AtmosphereSimulator::calculatePrimaryClimate(const SimulationParameters& params) {
    calculateTemperatures(params);
    calculateWinds(params);

    auto& cells = planet.getCells();
    float eff_sea = params.effective_sea_level();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    for (auto& cell : cells) {
        float px = static_cast<float>(cell.position.x);
        float py = static_cast<float>(cell.position.y);
        float pz = static_cast<float>(cell.position.z);
        float noise = rw_fbm3d(px * 2.0f, py * 2.0f, pz * 2.0f, 4, u_seed + 701u) * 15.0f;
        float lat_deg = std::abs(static_cast<float>(cell.latitude)) * 57.2957795f + noise;

        float itcz = std::exp(-0.5f * (lat_deg / 15.0f) * (lat_deg / 15.0f));
        float midlat = 0.75f * std::exp(-0.5f * ((lat_deg - 50.0f) / 15.0f) * ((lat_deg - 50.0f) / 15.0f));
        float subtrop = std::exp(-0.5f * ((lat_deg - 27.0f) / 10.0f) * ((lat_deg - 27.0f) / 10.0f));

        float base_m = std::max(0.10f, std::min(1.0f, (0.32f + 0.52f * (itcz + midlat)) * (1.0f - 0.60f * subtrop)));
        if (cell.elevation <= eff_sea) {
            cell.moisture = 0.85f;
            cell.precipitation = base_m;
        } else {
            cell.moisture = base_m * 0.65f;
            cell.precipitation = base_m * 0.65f;
        }
    }
}

void AtmosphereSimulator::calculateFullClimate(int moistureIterations, const SimulationParameters& params) {
    calculateTemperatures(params);
    calculateWinds(params);
    simulateMoisture(moistureIterations, params);
    assignBiomes(params);
}

void AtmosphereSimulator::calculateTemperatures(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    float lgm_offset = params.lgm_temp_anomaly / 55.0f;
    float effective_sea = params.effective_sea_level();
    const float obliquity_rad = 23.44f * static_cast<float>(Math::PI) / 180.0f;
    const float sin2_eps = std::sin(obliquity_rad) * std::sin(obliquity_rad);

    for (auto& cell : cells) {
        float px = static_cast<float>(cell.position.x);
        float py = static_cast<float>(cell.position.y);
        float pz = static_cast<float>(cell.position.z);

        // Ondulación de isotermas mediante ondas planetarias y ruido fractal (rompe el paralelismo zonal)
        float n_large = rw_fbm3d(px * 1.8f, py * 1.8f, pz * 1.8f, 4, u_seed + 501u);
        float n_med   = rw_fbm3d(px * 4.5f, py * 4.5f, pz * 4.5f, 3, u_seed + 509u);
        float phi = static_cast<float>(cell.latitude) + n_large * 0.11f + n_med * 0.04f;
        phi = std::max(-static_cast<float>(Math::PI) * 0.5f, std::min(static_cast<float>(Math::PI) * 0.5f, phi));

        float sin_phi = std::sin(phi);
        float p2 = 0.5f * (3.0f * sin_phi * sin_phi - 1.0f);
        float insolation = 1.0f - 0.5f * p2 * (1.0f - 1.5f * sin2_eps);

        // Rango calibrado para que los polos bajen hasta ~0.04 (-13°C) y formen hielo marino (< 0.20)
        // en ambos casquetes polares sin congelar las latitudes templadas.
        float baseTemp = (insolation - 0.46f) * 1.18f;

        float elevAboveSea = std::max(0.0f, cell.elevation - effective_sea);
        float lapseRate = elevAboveSea * 0.000088f;

        float temp = baseTemp - lapseRate + params.temp_offset + lgm_offset;

        if (cell.elevation <= effective_sea) {
            // Moderación marina en latitudes bajas/medias, pero permitiendo que el océano polar (>72° lat)
            // caiga por debajo del umbral de congelación (0.20) para formar banquisa de hielo sin tierra.
            float polar_chill = std::pow(std::abs(sin_phi), 4.0f) * 0.09f;
            temp = temp * 0.84f + 0.07f - polar_chill + cell.ocean_temp_anomaly * 1.35f;
            cell.temp_seasonality = 3.0f + 10.0f * std::abs(sin_phi);
        } else {
            // Variación térmica intracontinental + influencia costera de corrientes cálidas/frías
            temp += cell.ocean_temp_anomaly * 0.65f + n_med * 0.035f;
            cell.temp_seasonality = 6.0f + 32.0f * std::abs(sin_phi);
        }

        cell.temperature = std::max(0.0f, std::min(1.0f, temp));
    }

    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> tmp(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            float sum = cells[i].temperature * 2.0f;
            float count = 2.0f;
            for (size_t nid : cells[i].neighbors) {
                sum += cells[nid].temperature;
                count += 1.0f;
            }
            tmp[i] = sum / count;
        }
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i].temperature = tmp[i];
        }
    }
}

void AtmosphereSimulator::calculateWinds(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    int num_cells = static_cast<int>(cells.size());
    if (num_cells == 0) return;

    float eff_sea = params.effective_sea_level();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_lat(num_cells), h_elev(num_cells);
    std::vector<float> h_pressure(num_cells), h_eq_pressure(num_cells);
    std::vector<float> h_wind_u(num_cells, 0.0f), h_wind_v(num_cells, 0.0f), h_wind_w(num_cells, 0.0f);

    for (int i = 0; i < num_cells; ++i) {
        float px = static_cast<float>(cells[i].position.x);
        float py = static_cast<float>(cells[i].position.y);
        float pz = static_cast<float>(cells[i].position.z);
        h_pos_x[i] = px;
        h_pos_y[i] = py;
        h_pos_z[i] = pz;
        float phi = static_cast<float>(cells[i].latitude);
        h_lat[i] = phi;
        h_elev[i] = cells[i].elevation;

        // Perturbar la latitud de las bandas de presión con ruido sinóptico para crear centros
        // de alta y baja presión (ciclones y anticiclones) en vez de anillos rectos.
        float wave = rw_fbm3d(px * 1.6f, py * 1.6f, pz * 1.6f, 3, u_seed + 613u) * 0.22f;
        float p_zonal = 1.0f - 0.24f * std::cos(3.0f * (phi + wave));
        float land_thermal = (cells[i].elevation > eff_sea)
                                 ? -(cells[i].temperature - 0.50f) * 0.22f
                                 :  (0.50f - cells[i].temperature) * 0.11f;

        h_eq_pressure[i] = p_zonal + land_thermal;
        h_pressure[i] = h_eq_pressure[i];
    }

    const auto& h_neighbors = planet.getRowMajorNeighbors();
    const auto& h_num_neighbors = planet.getNeighborCounts();

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_lat, *d_elev;
    float *d_pressure, *d_next_pressure, *d_eq_pressure, *d_div;
    float *d_wind_u, *d_wind_v, *d_wind_w;
    int *d_neighbors, *d_num_neighbors;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lat, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pressure, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_pressure, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_eq_pressure, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_div, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lat, h_lat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_elev, h_elev.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pressure, h_pressure.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_eq_pressure, h_eq_pressure.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_div, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_wind_u, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_wind_v, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_wind_w, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    float dt = 0.04f;
    float grad_scale = 6.5f;

    for (int iter = 0; iter < params.swe_iterations; ++iter) {
        wind_update_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z, d_lat, d_elev,
            d_wind_u, d_wind_v, d_wind_w, d_pressure,
            d_neighbors, d_num_neighbors, eff_sea, dt, grad_scale
        );

        pressure_update_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z,
            d_wind_u, d_wind_v, d_wind_w,
            d_pressure, d_eq_pressure, d_next_pressure, d_div,
            d_neighbors, d_num_neighbors, dt
        );

        std::swap(d_pressure, d_next_pressure);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_wind_u.data(), d_wind_u, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wind_v.data(), d_wind_v, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wind_w.data(), d_wind_w, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    float max_speed = 1e-4f;
    for (int i = 0; i < num_cells; ++i) {
        float s = std::sqrt(h_wind_u[i] * h_wind_u[i] + h_wind_v[i] * h_wind_v[i] + h_wind_w[i] * h_wind_w[i]);
        if (s > max_speed) max_speed = s;
    }

    float inv_max = 1.0f / max_speed;
    for (int i = 0; i < num_cells; ++i) {
        cells[i].wind_velocity = Vector3(h_wind_u[i] * inv_max, h_wind_v[i] * inv_max, h_wind_w[i] * inv_max);
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_lat); cudaFree(d_elev);
    cudaFree(d_pressure); cudaFree(d_next_pressure); cudaFree(d_eq_pressure); cudaFree(d_div);
    cudaFree(d_wind_u); cudaFree(d_wind_v); cudaFree(d_wind_w);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
}

void AtmosphereSimulator::simulateMoisture(int iterations, const SimulationParameters& params) {
    auto& cells = planet.getCells();
    int num_cells = static_cast<int>(cells.size());
    if (num_cells == 0) return;

    float eff_sea = params.effective_sea_level();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_lat(num_cells), h_elev(num_cells), h_temp(num_cells), h_sst_anom(num_cells);
    std::vector<float> h_wind_u(num_cells), h_wind_v(num_cells), h_wind_w(num_cells);
    std::vector<float> h_wind_div(num_cells, 0.0f);
    std::vector<float> h_moisture(num_cells, 0.0f), h_precip(num_cells, 0.0f);

    for (int i = 0; i < num_cells; ++i) {
        float px = static_cast<float>(cells[i].position.x);
        float py = static_cast<float>(cells[i].position.y);
        float pz = static_cast<float>(cells[i].position.z);
        h_pos_x[i] = px;
        h_pos_y[i] = py;
        h_pos_z[i] = pz;
        h_lat[i] = static_cast<float>(cells[i].latitude);
        h_elev[i] = cells[i].elevation;
        h_temp[i] = cells[i].temperature;
        h_sst_anom[i] = cells[i].ocean_temp_anomaly;
        h_wind_u[i] = static_cast<float>(cells[i].wind_velocity.x);
        h_wind_v[i] = static_cast<float>(cells[i].wind_velocity.y);
        h_wind_w[i] = static_cast<float>(cells[i].wind_velocity.z);

        float q_cap = 0.42f + 0.58f * std::max(0.0f, std::min(1.0f, h_temp[i]));
        // En tierra empezamos con humedad baja uniforme modulada por ruido 3D (NO por bandas de latitud),
        // obligando a que toda la humedad continental provenga realmente del transporte de los vientos desde el mar.
        float init_noise = 0.5f + 0.5f * rw_fbm3d(px * 2.2f, py * 2.2f, pz * 2.2f, 3, u_seed + 733u);
        h_moisture[i] = (h_elev[i] <= eff_sea) ? (q_cap * 0.90f) : (q_cap * (0.18f + 0.22f * init_noise));
    }

    for (int i = 0; i < num_cells; ++i) {
        float div = 0.0f;
        for (size_t nid : cells[i].neighbors) {
            Vector3 d = Math::normalize(Vector3(
                cells[nid].position.x - cells[i].position.x,
                cells[nid].position.y - cells[i].position.y,
                cells[nid].position.z - cells[i].position.z
            ));
            double out_w = Math::dotProduct(cells[i].wind_velocity, d);
            double in_w  = -Math::dotProduct(cells[nid].wind_velocity, d);
            div += static_cast<float>(out_w - in_w);
        }
        if (!cells[i].neighbors.empty()) {
            div /= static_cast<float>(cells[i].neighbors.size());
        }
        // Dominio de la divergencia real del campo de vientos (75%) sobre el término zonal (25%)
        float wave = rw_fbm3d(h_pos_x[i] * 1.8f, h_pos_y[i] * 1.8f, h_pos_z[i] * 1.8f, 3, u_seed + 619u) * 0.25f;
        float hadley_div = -0.35f * std::cos(3.0f * (h_lat[i] + wave));
        h_wind_div[i] = 0.75f * div + 0.25f * hadley_div;
    }

    const auto& h_neighbors = planet.getRowMajorNeighbors();
    const auto& h_num_neighbors = planet.getNeighborCounts();

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_lat, *d_elev, *d_temp, *d_sst_anom;
    float *d_wind_u, *d_wind_v, *d_wind_w, *d_wind_div;
    float *d_moisture, *d_next_moisture, *d_precip;
    int *d_neighbors, *d_num_neighbors;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lat, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sst_anom, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_div, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_moisture, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_moisture, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_precip, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lat, h_lat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_elev, h_elev.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sst_anom, h_sst_anom.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_u, h_wind_u.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_v, h_wind_v.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_w, h_wind_w.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_div, h_wind_div.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_moisture, h_moisture.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_precip, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    int res_mult = 1 << std::max(0, params.subdivision_level - 5);
    int base_iters = std::max(28, iterations);
    int scaled_steps = base_iters * res_mult;
    float step_scale = 1.0f / static_cast<float>(res_mult);
    const float cfl_advect = 0.90f;

    for (int step = 0; step < scaled_steps; ++step) {
        moisture_advect_precip_kernel<<<numBlocks, blockSize>>>(
            num_cells,
            d_pos_x, d_pos_y, d_pos_z, d_lat,
            d_elev, d_temp, d_sst_anom,
            d_wind_u, d_wind_v, d_wind_w, d_wind_div,
            d_moisture, d_next_moisture, d_precip,
            d_neighbors, d_num_neighbors,
            eff_sea, cfl_advect, step_scale, u_seed
        );
        std::swap(d_moisture, d_next_moisture);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_moisture.data(), d_moisture, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_precip.data(), d_precip, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    float norm_scale = 1.0f / (static_cast<float>(base_iters) * 0.088f);
    for (int i = 0; i < num_cells; ++i) {
        cells[i].moisture = std::max(0.0f, std::min(1.0f, h_moisture[i]));
        cells[i].precipitation = std::max(0.0f, std::min(1.0f, h_precip[i] * norm_scale));
    }

    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> tmp(num_cells);
        for (int i = 0; i < num_cells; ++i) {
            float sum = 0.0f;
            for (size_t nid : cells[i].neighbors) sum += cells[nid].precipitation;
            float nbrMean = cells[i].neighbors.empty() ? cells[i].precipitation : (sum / cells[i].neighbors.size());
            tmp[i] = 0.60f * cells[i].precipitation + 0.40f * nbrMean;
        }
        for (int i = 0; i < num_cells; ++i) {
            cells[i].precipitation = tmp[i];
        }
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z); cudaFree(d_lat);
    cudaFree(d_elev); cudaFree(d_temp); cudaFree(d_sst_anom);
    cudaFree(d_wind_u); cudaFree(d_wind_v); cudaFree(d_wind_w); cudaFree(d_wind_div);
    cudaFree(d_moisture); cudaFree(d_next_moisture); cudaFree(d_precip);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
}

void AtmosphereSimulator::assignBiomes(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    float eff_sea = params.effective_sea_level();
    const uint32_t u_seed = static_cast<uint32_t>(params.seed);

    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= eff_sea) {
            cells[i].biome = BiomeType::OCEAN;
            continue;
        }
        // Pequeña perturbación ecotonal (ecotone jitter) para que las fronteras entre biomas
        // sean orgánicas y entrelazadas como en mapas ecológicos reales.
        float px = static_cast<float>(cells[i].position.x);
        float py = static_cast<float>(cells[i].position.y);
        float pz = static_cast<float>(cells[i].position.z);
        float ecotone = rw_fbm3d(px * 9.0f, py * 9.0f, pz * 9.0f, 3, u_seed + 991u);

        float temp_c = (cells[i].temperature * 55.0f - 15.0f) + ecotone * 1.4f;
        float precip = std::max(0.0f, std::min(1.0f, cells[i].precipitation + ecotone * 0.035f));
        cells[i].biome = classifyBiome(temp_c, precip, cells[i].temp_seasonality);
    }

    std::vector<BiomeType> next_biomes(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation <= eff_sea) {
            next_biomes[i] = BiomeType::OCEAN;
            continue;
        }

        int counts[static_cast<int>(BiomeType::THORN_SCRUB) + 1] = {0};
        counts[static_cast<int>(cells[i].biome)] = 2;

        for (size_t nid : cells[i].neighbors) {
            if (cells[nid].elevation > eff_sea) {
                counts[static_cast<int>(cells[nid].biome)]++;
            }
        }

        int max_count = 0;
        BiomeType most_common = cells[i].biome;
        for (int b = 0; b <= static_cast<int>(BiomeType::THORN_SCRUB); ++b) {
            if (counts[b] > max_count) {
                max_count = counts[b];
                most_common = static_cast<BiomeType>(b);
            }
        }
        next_biomes[i] = most_common;
    }

    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].elevation > eff_sea) {
            cells[i].biome = next_biomes[i];
        }
    }
}

} // namespace Ravis