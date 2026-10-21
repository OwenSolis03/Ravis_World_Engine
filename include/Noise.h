#pragma once
// ---------------------------------------------------------------------------
// Coherent 3D value noise with deterministic seed support and inter-octave
// domain rotation, shared by host (C++) and device (CUDA) code.
// ---------------------------------------------------------------------------

#include <cstdint>

#ifdef __CUDACC__
  #define RW_HD __host__ __device__
#else
  #define RW_HD
  #include <cmath>
#endif

namespace Ravis {

RW_HD inline uint32_t rw_ihash(int x, int y, int z, uint32_t seed = 0u) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u
               + static_cast<uint32_t>(y) * 668265263u
               + static_cast<uint32_t>(z) * 2246822519u
               + seed * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

// Valor de celosía hasheado en [0, 1).
RW_HD inline float rw_hashf(int x, int y, int z, uint32_t seed = 0u) {
    return (rw_ihash(x, y, z, seed) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

RW_HD inline float rw_lerp(float a, float b, float t) { return a + t * (b - a); }

// Rotación ortogonal 3D entre octavas para romper la alineación con los ejes X, Y, Z.
RW_HD inline void rw_rotate_octave(float& x, float& y, float& z) {
    const float rx =  0.36f * x - 0.48f * y + 0.80f * z;
    const float ry =  0.80f * x + 0.60f * y + 0.00f * z;
    const float rz = -0.48f * x + 0.64f * y + 0.60f * z;
    x = rx; y = ry; z = rz;
}

// Ruido de valor en [-1, 1].
RW_HD inline float rw_value_noise3d(float x, float y, float z, uint32_t seed = 0u) {
    // Desplazamiento irracional + variación determinista por semilla
    float sx = 17.13f + static_cast<float>((seed * 1664525u + 1013904223u) & 0xFFFFu) * 0.0037f;
    float sy = 71.37f + static_cast<float>((seed * 22695477u + 1u) & 0xFFFFu) * 0.0053f;
    float sz = 39.71f + static_cast<float>((seed * 1103515245u + 12345u) & 0xFFFFu) * 0.0029f;
    x += sx; y += sy; z += sz;

#ifdef __CUDACC__
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
#else
    float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
#endif
    int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
    float tx = x - fx, ty = y - fy, tz = z - fz;

    // Curva quíntica C2 (smootherstep).
    float ux = tx * tx * tx * (tx * (tx * 6.0f - 15.0f) + 10.0f);
    float uy = ty * ty * ty * (ty * (ty * 6.0f - 15.0f) + 10.0f);
    float uz = tz * tz * tz * (tz * (tz * 6.0f - 15.0f) + 10.0f);

    float c000 = rw_hashf(ix,     iy,     iz,     seed);
    float c100 = rw_hashf(ix + 1, iy,     iz,     seed);
    float c010 = rw_hashf(ix,     iy + 1, iz,     seed);
    float c110 = rw_hashf(ix + 1, iy + 1, iz,     seed);
    float c001 = rw_hashf(ix,     iy,     iz + 1, seed);
    float c101 = rw_hashf(ix + 1, iy,     iz + 1, seed);
    float c011 = rw_hashf(ix,     iy + 1, iz + 1, seed);
    float c111 = rw_hashf(ix + 1, iy + 1, iz + 1, seed);

    float x00 = rw_lerp(c000, c100, ux);
    float x10 = rw_lerp(c010, c110, ux);
    float x01 = rw_lerp(c001, c101, ux);
    float x11 = rw_lerp(c011, c111, ux);
    float y0  = rw_lerp(x00, x10, uy);
    float y1  = rw_lerp(x01, x11, uy);
    return rw_lerp(y0, y1, uz) * 2.0f - 1.0f;
}

// Movimiento Browniano Fractal (fBm) con rotación de octavas, resultado en [-1, 1].
RW_HD inline float rw_fbm3d(float x, float y, float z, int octaves, uint32_t seed = 0u) {
    float value = 0.0f, amp = 1.0f, total = 0.0f;
    float px = x, py = y, pz = z;
    for (int i = 0; i < octaves; ++i) {
        value += rw_value_noise3d(px, py, pz, seed + static_cast<uint32_t>(i) * 101u) * amp;
        total += amp;
        amp *= 0.5f;
        rw_rotate_octave(px, py, pz);
        px *= 2.02f; py *= 2.02f; pz *= 2.02f;
    }
    return total > 0.0f ? (value / total) : 0.0f;
}

// Ruido de crestas multifractal (Ridge noise) con amortiguamiento por elevación previo, resultado en [0, 1].
RW_HD inline float rw_ridge3d(float x, float y, float z, int octaves, uint32_t seed = 0u) {
    float value = 0.0f, amp = 1.0f, weight = 1.0f, total = 0.0f;
    float px = x, py = y, pz = z;
    for (int i = 0; i < octaves; ++i) {
        float n = rw_value_noise3d(px, py, pz, seed + static_cast<uint32_t>(i) * 197u);
        n = 1.0f - (n < 0.0f ? -n : n);
        n = n * n * weight;
        weight = n < 0.0f ? 0.0f : (n > 1.0f ? 1.0f : n);
        value += n * amp;
        total += amp;
        amp *= 0.5f;
        rw_rotate_octave(px, py, pz);
        px *= 2.03f; py *= 2.03f; pz *= 2.03f;
    }
    return total > 0.0f ? (value / total) : 0.0f;
}

} // namespace Ravis