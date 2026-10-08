#pragma once
// ---------------------------------------------------------------------------
// Coherent 3D value noise, shared by host (MSVC) and device (nvcc) code.
//
// Replaces the previous "sin of a dot product" hash noise, which was not
// spatially coherent: sampled over a sphere it produced visible hexagonal /
// banded interference lattices (angular coastlines, hex-shaped inland seas,
// dead-straight plate boundaries). Value noise interpolates a hashed integer
// lattice with a quintic fade, so it varies smoothly and has no preferred
// direction.
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
  #define RW_HD __host__ __device__
#else
  #define RW_HD
  #include <cmath>
#endif

namespace Ravis {

RW_HD inline unsigned int rw_ihash(int x, int y, int z) {
    unsigned int h = (unsigned int)x * 374761393u
                   + (unsigned int)y * 668265263u
                   + (unsigned int)z * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

// Hashed lattice value in [0, 1).
RW_HD inline float rw_hashf(int x, int y, int z) {
    return (rw_ihash(x, y, z) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

RW_HD inline float rw_lerp(float a, float b, float t) { return a + t * (b - a); }

// Value noise in [-1, 1].
RW_HD inline float rw_value_noise3d(float x, float y, float z) {
    // Irrational offset: keep the lattice origin off (0,0,0) so the sphere
    // centre is not a degenerate cell corner and there is no symmetry through
    // the origin.
    x += 17.13f; y += 71.37f; z += 39.71f;

#ifdef __CUDACC__
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
#else
    float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
#endif
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = x - fx, ty = y - fy, tz = z - fz;

    // Quintic smootherstep fade.
    float ux = tx * tx * tx * (tx * (tx * 6.0f - 15.0f) + 10.0f);
    float uy = ty * ty * ty * (ty * (ty * 6.0f - 15.0f) + 10.0f);
    float uz = tz * tz * tz * (tz * (tz * 6.0f - 15.0f) + 10.0f);

    float c000 = rw_hashf(ix,     iy,     iz    );
    float c100 = rw_hashf(ix + 1, iy,     iz    );
    float c010 = rw_hashf(ix,     iy + 1, iz    );
    float c110 = rw_hashf(ix + 1, iy + 1, iz    );
    float c001 = rw_hashf(ix,     iy,     iz + 1);
    float c101 = rw_hashf(ix + 1, iy,     iz + 1);
    float c011 = rw_hashf(ix,     iy + 1, iz + 1);
    float c111 = rw_hashf(ix + 1, iy + 1, iz + 1);

    float x00 = rw_lerp(c000, c100, ux);
    float x10 = rw_lerp(c010, c110, ux);
    float x01 = rw_lerp(c001, c101, ux);
    float x11 = rw_lerp(c011, c111, ux);
    float y0  = rw_lerp(x00, x10, uy);
    float y1  = rw_lerp(x01, x11, uy);
    return rw_lerp(y0, y1, uz) * 2.0f - 1.0f;
}

// Fractal Brownian Motion — multi-octave value noise, result in [-1, 1].
RW_HD inline float rw_fbm3d(float x, float y, float z, int octaves) {
    float value = 0.0f, amp = 1.0f, freq = 1.0f, total = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value += rw_value_noise3d(x * freq, y * freq, z * freq) * amp;
        total += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return value / total;
}

// Ridge noise (1 - |noise|, sharpened), result in [0, 1].
RW_HD inline float rw_ridge3d(float x, float y, float z, int octaves) {
    float value = 0.0f, amp = 1.0f, freq = 1.0f, total = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        float n = rw_value_noise3d(x * freq, y * freq, z * freq);
        n = 1.0f - (n < 0.0f ? -n : n);
        n *= n;
        value += n * amp;
        total += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return value / total;
}

} // namespace Ravis
