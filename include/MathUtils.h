#pragma once

#include "PlanetData.h"
#include <algorithm>
#include <cmath>

namespace Ravis {
namespace Math {

constexpr double PI = 3.14159265358979323846;
constexpr double PLANET_RADIUS_M = 6371000.0; // Radio terrestre medio en metros (SI)

// Convierte coordenadas esféricas (latitud, longitud en radianes) a cartesianas 3D en la esfera unitaria.
inline Vector3 sphericalToCartesian(double lat_rad, double lon_rad) {
    return Vector3(
        std::cos(lat_rad) * std::cos(lon_rad),
        std::cos(lat_rad) * std::sin(lon_rad),
        std::sin(lat_rad)
    );
}

// Convierte coordenadas cartesianas 3D en la esfera unitaria a esféricas (radianes) con clamp seguro.
inline void cartesianToSpherical(const Vector3& pos, double& out_lat, double& out_lon) {
    double clamped_z = std::max(-1.0, std::min(1.0, pos.z));
    out_lat = std::asin(clamped_z);
    out_lon = std::atan2(pos.y, pos.x);
}

// Normaliza un vector 3D
inline Vector3 normalize(const Vector3& v) {
    double length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 1e-12) {
        return Vector3(v.x / length, v.y / length, v.z / length);
    }
    return Vector3(0.0, 0.0, 0.0);
}

// Distancia euclidiana (cuerda) entre dos puntos 3D
inline double distance(const Vector3& a, const Vector3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Producto punto entre dos vectores 3D
inline double dotProduct(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Distancia geodésica (ángulo central en radianes) sobre la esfera unitaria
inline double geodesicDistanceRad(const Vector3& a, const Vector3& b) {
    double d = std::max(-1.0, std::min(1.0, dotProduct(a, b)));
    return std::acos(d);
}

// Producto cruz entre dos vectores 3D
inline Vector3 crossProduct(const Vector3& a, const Vector3& b) {
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Construye la base tangente ortonormal local (Este, Norte) en un punto de la esfera unitaria.
inline void localTangentBasis(const Vector3& unit_pos, Vector3& out_east, Vector3& out_north) {
    // Eje polar Norte = (0, 0, 1). Este = normalize(Z x pos) = normalize(-pos.y, pos.x, 0)
    double xy_len = std::sqrt(unit_pos.x * unit_pos.x + unit_pos.y * unit_pos.y);
    if (xy_len > 1e-6) {
        out_east = Vector3(-unit_pos.y / xy_len, unit_pos.x / xy_len, 0.0);
        out_north = crossProduct(unit_pos, out_east);
    } else {
        // Singularidad en los polos exactos
        out_east = Vector3(1.0, 0.0, 0.0);
        out_north = Vector3(0.0, unit_pos.z >= 0.0 ? 1.0 : -1.0, 0.0);
    }
}

// Fórmula de rotación de Rodrigues para rotar el vector v alrededor del eje unitario k por un ángulo theta
inline Vector3 rodriguesRotation(const Vector3& v, const Vector3& k, double theta) {
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    Vector3 cross = crossProduct(k, v);
    double dot = dotProduct(k, v);

    return Vector3(
        v.x * cos_theta + cross.x * sin_theta + k.x * dot * (1.0 - cos_theta),
        v.y * cos_theta + cross.y * sin_theta + k.y * dot * (1.0 - cos_theta),
        v.z * cos_theta + cross.z * sin_theta + k.z * dot * (1.0 - cos_theta)
    );
}

} // namespace Math
} // namespace Ravis