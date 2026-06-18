#pragma once

#include "PlanetData.h"
#include <cmath>

namespace Ravis {
namespace Math {

constexpr double PI = 3.14159265358979323846;

// Convert spherical coordinates (latitude, longitude in radians) to 3D Cartesian coordinates on a unit sphere.
inline Vector3 sphericalToCartesian(double lat_rad, double lon_rad) {
    return Vector3(
        std::cos(lat_rad) * std::cos(lon_rad),
        std::cos(lat_rad) * std::sin(lon_rad),
        std::sin(lat_rad)
    );
}

// Convert 3D Cartesian coordinates on a unit sphere to spherical coordinates (latitude, longitude in radians).
inline void cartesianToSpherical(const Vector3& pos, double& out_lat, double& out_lon) {
    out_lat = std::asin(pos.z);
    out_lon = std::atan2(pos.y, pos.x);
}

// Normalize a 3D vector
inline Vector3 normalize(const Vector3& v) {
    double length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0) {
        return Vector3(v.x / length, v.y / length, v.z / length);
    }
    return v;
}

// Distance between two 3D points
inline double distance(const Vector3& a, const Vector3& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace Math
} // namespace Ravis
