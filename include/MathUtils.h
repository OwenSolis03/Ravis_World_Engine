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

// Dot product of two 3D vectors
inline double dotProduct(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product of two 3D vectors
inline Vector3 crossProduct(const Vector3& a, const Vector3& b) {
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Rodrigues' rotation formula to rotate vector v around unit axis k by angle theta
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
