#include "../include/OceanSimulator.h"
#include "../include/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace Ravis {

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================================
// Kernels CUDA para Dinámica Oceánica Esférica
// ============================================================================

// Inicializa el forzamiento de Ekman sobre el océano a partir del viento superficial
// (o del patrón planetario de 3 celdas si el viento aún es cero).
__global__ void ocean_init_forcing_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* lat,
    const float* elevation,
    const float* wind_u, const float* wind_v, const float* wind_w,
    float* cur_u, float* cur_v, float* cur_w,
    float sea_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    if (elevation[i] > sea_level) {
        cur_u[i] = 0.0f;
        cur_v[i] = 0.0f;
        cur_w[i] = 0.0f;
        return;
    }

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float phi = lat[i];

    // Base tangente local (Este, Norte) en la esfera
    float xy = sqrtf(px * px + py * py);
    float east_x = (xy > 1e-5f) ? (-py / xy) : 1.0f;
    float east_y = (xy > 1e-5f) ? ( px / xy) : 0.0f;
    float east_z = 0.0f;

    float north_x = py * east_z - pz * east_y;
    float north_y = pz * east_x - px * east_z;
    float north_z = px * east_y - py * east_x;

    float wu = wind_u[i], wv = wind_v[i], ww = wind_w[i];
    float w_mag = sqrtf(wu * wu + wv * wv + ww * ww);

    // Si aún no se ha ejecutado calculateWinds(), usar perfil analítico de 3 celdas
    if (w_mag < 1e-4f) {
        // Zonal: Alisios del Este (-cos) en 0..30°, Westerlies del Oeste en 30..60°, Polares del Este en 60..90°
        float u_zonal = -cosf(3.0f * phi) * 0.75f;
        float v_merid = -sinf(3.0f * phi) * 0.25f * (phi >= 0.0f ? 1.0f : -1.0f);
        wu = u_zonal * east_x + v_merid * north_x;
        wv = u_zonal * east_y + v_merid * north_y;
        ww = u_zonal * east_z + v_merid * north_z;
    }

    // Deflexión de Ekman (~35° a la derecha en Hemisferio Norte, izquierda en Hemisferio Sur)
    float sgn = (phi >= 0.0f) ? 1.0f : -1.0f;
    float sin_lat = fabsf(sinf(phi));
    float ekman_angle = sgn * (0.55f * sin_lat); // disminuye cerca del ecuador
    float ca = cosf(ekman_angle), sa = sinf(ekman_angle);

    // Rotación de vector tangente alrededor de la normal exterior p: v_rot = cos*w - sin*(p x w)
    float pxw_x = py * ww - pz * wv;
    float pxw_y = pz * wu - px * ww;
    float pxw_z = px * wv - py * wu;

    cur_u[i] = ca * wu - sa * pxw_x;
    cur_v[i] = ca * wv - sa * pxw_y;
    cur_w[i] = ca * ww - sa * pxw_z;
}

// Relaja el campo de corrientes oceánicas respetando las barreras continentales,
// imponiendo condición de deslizamiento libre paralelo a la costa y aplicando
// intensificación de borde oeste (efecto beta de Munk: df/dy = 2*Omega*cos(lat)/R).
__global__ void ocean_gyre_relax_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* lat,
    const float* elevation,
    const float* in_u, const float* in_v, const float* in_w,
    float* out_u, float* out_v, float* out_w,
    const int* neighbors, const int* num_neighbors,
    float sea_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    if (elevation[i] > sea_level) {
        out_u[i] = 0.0f;
        out_v[i] = 0.0f;
        out_w[i] = 0.0f;
        return;
    }

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float phi = lat[i];
    float beta = cosf(phi); // Derivada planetaria de Coriolis

    // Vector unitario Este local para medir correctamente el desplazamiento zonal
    float xy = sqrtf(px * px + py * py);
    float east_x = (xy > 1e-5f) ? (-py / xy) : 1.0f;
    float east_y = (xy > 1e-5f) ? ( px / xy) : 0.0f;

    float sum_u = in_u[i] * 1.2f;
    float sum_v = in_v[i] * 1.2f;
    float sum_w = in_w[i] * 1.2f;
    float weight_sum = 1.2f;

    // Acumulador de normal costera (apuntando hacia el océano desde tierra)
    float coast_nx = 0.0f, coast_ny = 0.0f, coast_nz = 0.0f;
    bool near_coast = false;

    int nn = num_neighbors[i];
    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        dx /= dist; dy /= dist; dz /= dist;

        if (elevation[nid] <= sea_level) {
            // Proyección zonal sobre el vector Este local (positivo = vecino al Este, negativo = al Oeste)
            float zonal_step = dx * east_x + dy * east_y;
            // Intensificación del borde oeste (Munk): mayor peso acoplado en márgenes occidentales
            float w = 1.0f - 0.45f * zonal_step * beta;
            w = fmaxf(0.25f, fminf(1.85f, w));

            sum_u += in_u[nid] * w;
            sum_v += in_v[nid] * w;
            sum_w += in_w[nid] * w;
            weight_sum += w;
        } else {
            // El vecino es continente: acumular vector alejándose de la costa
            near_coast = true;
            coast_nx -= dx;
            coast_ny -= dy;
            coast_nz -= dz;
        }
    }

    float u = sum_u / weight_sum;
    float v = sum_v / weight_sum;
    float w = sum_w / weight_sum;

    // Si estamos junto a una costa, eliminar la componente normal hacia tierra (free-slip boundary)
    // y conservar la energía cinética redirigiéndola paralelamente al margen continental.
    if (near_coast) {
        float cn_dot_p = coast_nx * px + coast_ny * py + coast_nz * pz;
        coast_nx -= cn_dot_p * px;
        coast_ny -= cn_dot_p * py;
        coast_nz -= cn_dot_p * pz;
        float cn_len = sqrtf(coast_nx * coast_nx + coast_ny * coast_ny + coast_nz * coast_nz);
        if (cn_len > 1e-5f) {
            coast_nx /= cn_len; coast_ny /= cn_len; coast_nz /= cn_len;
            float speed_before = sqrtf(u * u + v * v + w * w);
            float into_land = u * coast_nx + v * coast_ny + w * coast_nz;
            if (into_land < 0.0f) {
                u -= into_land * coast_nx;
                v -= into_land * coast_ny;
                w -= into_land * coast_nz;
                // Conservar velocidad a lo largo de la corriente de borde costero
                float speed_after = sqrtf(u * u + v * v + w * w);
                if (speed_after > 1e-5f) {
                    float boost = (speed_before / speed_after) * 1.08f;
                    u *= boost; v *= boost; w *= boost;
                }
            }
        }
    }

    // Proyectar al plano tangente de la esfera y acotar magnitud
    float dot_p = u * px + v * py + w * pz;
    u -= dot_p * px;
    v -= dot_p * py;
    w -= dot_p * pz;

    float mag = sqrtf(u * u + v * v + w * w);
    if (mag > 1.25f) {
        float s = 1.25f / mag;
        u *= s; v *= s; w *= s;
    }

    out_u[i] = u;
    out_v[i] = v;
    out_w[i] = w;
}

// Advecta la temperatura superficial del mar (SST) a lo largo de las corrientes
// y calcula el enfriamiento por surgencia costera (coastal upwelling) donde la
// corriente diverge de la costa.
__global__ void ocean_sst_advection_kernel(
    int num_cells,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* elevation,
    const float* cur_u, const float* cur_v, const float* cur_w,
    const float* temp_in, float* temp_out,
    float* sst_anomaly,
    const int* neighbors, const int* num_neighbors,
    float sea_level)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    int nn = num_neighbors[i];

    // En tierra firme: difundir suavemente la influencia marítima en la franja costera
    if (elevation[i] > sea_level) {
        float sum_anom = 0.0f;
        int count_ocean = 0;
        for (int n = 0; n < nn; ++n) {
            int nid = neighbors[i * 6 + n];
            if (elevation[nid] <= sea_level) {
                sum_anom += sst_anomaly[nid];
                count_ocean++;
            }
        }
        if (count_ocean > 0) {
            float coastal_mod = (sum_anom / static_cast<float>(count_ocean)) * 0.45f;
            sst_anomaly[i] = coastal_mod;
            temp_out[i] = fmaxf(0.0f, fminf(1.0f, temp_in[i] + coastal_mod * 0.25f));
        } else {
            temp_out[i] = temp_in[i];
        }
        return;
    }

    // Celda oceánica: advección aguas arriba (upwind) + difusión térmica horizontal
    float u = cur_u[i], v = cur_v[i], w = cur_w[i];
    float upwind_temp = temp_in[i];
    float upwind_weight = 1.0f;
    float mean_temp = temp_in[i];
    int ocean_nbrs = 1;

    float offshore_divergence = 0.0f;
    bool is_coastal_ocean = false;

    for (int n = 0; n < nn; ++n) {
        int nid = neighbors[i * 6 + n];
        float dx = pos_x[nid] - px;
        float dy = pos_y[nid] - py;
        float dz = pos_z[nid] - pz;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
        dx /= dist; dy /= dist; dz /= dist;

        if (elevation[nid] <= sea_level) {
            mean_temp += temp_in[nid];
            ocean_nbrs++;

            // Si el vecino está aguas arriba (dirección opuesta a mi corriente), recibimos su calor
            float align = -(dx * u + dy * v + dz * w);
            if (align > 0.0f) {
                upwind_temp += temp_in[nid] * align * 2.2f;
                upwind_weight += align * 2.2f;
            }
        } else {
            is_coastal_ocean = true;
            // Si la corriente se aleja del continente vecino (dx apunta a tierra), hay surgencia de agua fría profunda
            float away_from_land = -(dx * u + dy * v + dz * w);
            if (away_from_land > 0.0f) {
                offshore_divergence += away_from_land;
            }
        }
    }

    float advected = upwind_temp / upwind_weight;
    float diffused = mean_temp / static_cast<float>(ocean_nbrs);
    float new_temp = 0.72f * advected + 0.28f * diffused;

    // Enfriamiento por surgencia costera (Upwelling en bordes orientales subtropicales)
    if (is_coastal_ocean && offshore_divergence > 0.0f) {
        new_temp -= fminf(0.06f, offshore_divergence * 0.035f);
    }

    new_temp = fmaxf(0.0f, fminf(1.0f, new_temp));
    sst_anomaly[i] += (new_temp - temp_in[i]);
    sst_anomaly[i] = fmaxf(-0.18f, fminf(0.18f, sst_anomaly[i]));
    temp_out[i] = new_temp;
}

// ============================================================================
// Métodos en Host (CPU)
// ============================================================================

OceanSimulator::OceanSimulator(GoldbergPolyhedron& p) : planet(p) {}

void OceanSimulator::simulateCurrents(const SimulationParameters& params) {
    auto& cells = planet.getCells();
    int num_cells = static_cast<int>(cells.size());
    if (num_cells == 0) return;

    float eff_sea = params.effective_sea_level();

    std::vector<float> h_pos_x(num_cells), h_pos_y(num_cells), h_pos_z(num_cells);
    std::vector<float> h_lat(num_cells), h_elev(num_cells), h_temp(num_cells);
    std::vector<float> h_wind_u(num_cells), h_wind_v(num_cells), h_wind_w(num_cells);
    std::vector<float> h_cur_u(num_cells, 0.0f), h_cur_v(num_cells, 0.0f), h_cur_w(num_cells, 0.0f);
    std::vector<float> h_sst_anom(num_cells, 0.0f);

    for (int i = 0; i < num_cells; ++i) {
        h_pos_x[i] = static_cast<float>(cells[i].position.x);
        h_pos_y[i] = static_cast<float>(cells[i].position.y);
        h_pos_z[i] = static_cast<float>(cells[i].position.z);
        h_lat[i] = static_cast<float>(cells[i].latitude);
        h_elev[i] = cells[i].elevation;
        h_temp[i] = cells[i].temperature;
        h_wind_u[i] = static_cast<float>(cells[i].wind_velocity.x);
        h_wind_v[i] = static_cast<float>(cells[i].wind_velocity.y);
        h_wind_w[i] = static_cast<float>(cells[i].wind_velocity.z);
    }

    const auto& h_neighbors = planet.getRowMajorNeighbors();
    const auto& h_num_neighbors = planet.getNeighborCounts();

    float *d_pos_x, *d_pos_y, *d_pos_z, *d_lat, *d_elev;
    float *d_wind_u, *d_wind_v, *d_wind_w;
    float *d_cur_u, *d_cur_v, *d_cur_w, *d_next_u, *d_next_v, *d_next_w;
    float *d_temp, *d_next_temp, *d_sst_anom;
    int *d_neighbors, *d_num_neighbors;

    CHECK_CUDA(cudaMalloc(&d_pos_x, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_y, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pos_z, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lat, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elev, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wind_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cur_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cur_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cur_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_u, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_v, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_w, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_temp, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sst_anom, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, num_cells * 6 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_neighbors, num_cells * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pos_x, h_pos_x.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_y, h_pos_y.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos_z, h_pos_z.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lat, h_lat.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_elev, h_elev.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_u, h_wind_u.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_v, h_wind_v.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wind_w, h_wind_w.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_temp, h_temp.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sst_anom, 0, num_cells * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors.data(), num_cells * 6 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_num_neighbors, h_num_neighbors.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_cells + blockSize - 1) / blockSize;

    // 1. Inicializar transporte de Ekman
    ocean_init_forcing_kernel<<<numBlocks, blockSize>>>(
        num_cells, d_pos_x, d_pos_y, d_pos_z, d_lat, d_elev,
        d_wind_u, d_wind_v, d_wind_w,
        d_cur_u, d_cur_v, d_cur_w, eff_sea
    );

    // 2. Relajar giros oceánicos (Munk + bloqueo continental)
    for (int iter = 0; iter < 30; ++iter) {
        ocean_gyre_relax_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z, d_lat, d_elev,
            d_cur_u, d_cur_v, d_cur_w,
            d_next_u, d_next_v, d_next_w,
            d_neighbors, d_num_neighbors, eff_sea
        );
        std::swap(d_cur_u, d_next_u);
        std::swap(d_cur_v, d_next_v);
        std::swap(d_cur_w, d_next_w);
    }

    // 3. Advectar temperatura superficial del mar (SST) y surgencia costera
    for (int iter = 0; iter < 18; ++iter) {
        ocean_sst_advection_kernel<<<numBlocks, blockSize>>>(
            num_cells, d_pos_x, d_pos_y, d_pos_z, d_elev,
            d_cur_u, d_cur_v, d_cur_w,
            d_temp, d_next_temp, d_sst_anom,
            d_neighbors, d_num_neighbors, eff_sea
        );
        std::swap(d_temp, d_next_temp);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_cur_u.data(), d_cur_u, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cur_v.data(), d_cur_v, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cur_w.data(), d_cur_w, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_temp.data(), d_temp, num_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sst_anom.data(), d_sst_anom, num_cells * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_cells; ++i) {
        cells[i].ocean_current = Vector3(h_cur_u[i], h_cur_v[i], h_cur_w[i]);
        cells[i].temperature = h_temp[i];
        cells[i].ocean_temp_anomaly = h_sst_anom[i];
    }

    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_lat); cudaFree(d_elev);
    cudaFree(d_wind_u); cudaFree(d_wind_v); cudaFree(d_wind_w);
    cudaFree(d_cur_u); cudaFree(d_cur_v); cudaFree(d_cur_w);
    cudaFree(d_next_u); cudaFree(d_next_v); cudaFree(d_next_w);
    cudaFree(d_temp); cudaFree(d_next_temp); cudaFree(d_sst_anom);
    cudaFree(d_neighbors); cudaFree(d_num_neighbors);
}

} // namespace Ravis