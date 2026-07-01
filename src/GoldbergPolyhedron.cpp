#include "../include/GoldbergPolyhedron.h"
#include "../include/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace Ravis {

GoldbergPolyhedron::GoldbergPolyhedron() {}

size_t GoldbergPolyhedron::addVertex(const Vector3& v) {
    Vector3 normalized = Math::normalize(v);
    vertices.push_back(normalized);
    return vertices.size() - 1;
}

size_t GoldbergPolyhedron::getMidpointIndex(size_t v1, size_t v2) {
    // Ensure smaller index is first to maintain consistency
    if (v1 > v2) std::swap(v1, v2);
    
    auto key = std::make_pair(v1, v2);
    auto it = midpointCache.find(key);
    if (it != midpointCache.end()) {
        return it->second;
    }

    // Calculate midpoint
    Vector3 p1 = vertices[v1];
    Vector3 p2 = vertices[v2];
    Vector3 mid(
        (p1.x + p2.x) / 2.0,
        (p1.y + p2.y) / 2.0,
        (p1.z + p2.z) / 2.0
    );

    size_t index = addVertex(mid);
    midpointCache[key] = index;
    return index;
}

void GoldbergPolyhedron::buildBaseIcosahedron() {
    vertices.clear();
    triangles.clear();

    const double t = (1.0 + std::sqrt(5.0)) / 2.0;

    addVertex(Vector3(-1,  t,  0));
    addVertex(Vector3( 1,  t,  0));
    addVertex(Vector3(-1, -t,  0));
    addVertex(Vector3( 1, -t,  0));

    addVertex(Vector3( 0, -1,  t));
    addVertex(Vector3( 0,  1,  t));
    addVertex(Vector3( 0, -1, -t));
    addVertex(Vector3( 0,  1, -t));

    addVertex(Vector3( t,  0, -1));
    addVertex(Vector3( t,  0,  1));
    addVertex(Vector3(-t,  0, -1));
    addVertex(Vector3(-t,  0,  1));

    // 5 faces around point 0
    triangles.emplace_back(0, 11, 5);
    triangles.emplace_back(0, 5, 1);
    triangles.emplace_back(0, 1, 7);
    triangles.emplace_back(0, 7, 10);
    triangles.emplace_back(0, 10, 11);

    // 5 adjacent faces
    triangles.emplace_back(1, 5, 9);
    triangles.emplace_back(5, 11, 4);
    triangles.emplace_back(11, 10, 2);
    triangles.emplace_back(10, 7, 6);
    triangles.emplace_back(7, 1, 8);

    // 5 faces around point 3
    triangles.emplace_back(3, 9, 4);
    triangles.emplace_back(3, 4, 2);
    triangles.emplace_back(3, 2, 6);
    triangles.emplace_back(3, 6, 8);
    triangles.emplace_back(3, 8, 9);

    // 5 adjacent faces
    triangles.emplace_back(4, 9, 5);
    triangles.emplace_back(2, 4, 11);
    triangles.emplace_back(6, 2, 10);
    triangles.emplace_back(8, 6, 7);
    triangles.emplace_back(9, 8, 1);
}

void GoldbergPolyhedron::subdivide() {
    std::vector<Triangle> newTriangles;
    midpointCache.clear();

    for (const auto& tri : triangles) {
        size_t a = getMidpointIndex(tri.v[0], tri.v[1]);
        size_t b = getMidpointIndex(tri.v[1], tri.v[2]);
        size_t c = getMidpointIndex(tri.v[2], tri.v[0]);

        newTriangles.emplace_back(tri.v[0], a, c);
        newTriangles.emplace_back(tri.v[1], b, a);
        newTriangles.emplace_back(tri.v[2], c, b);
        newTriangles.emplace_back(a, b, c);
    }
    triangles = newTriangles;
}

void GoldbergPolyhedron::extractCells() {
    cells.clear();
    cells.resize(vertices.size());

    for (size_t i = 0; i < vertices.size(); ++i) {
        cells[i].id = i;
        cells[i].position = vertices[i];
        
        double lat, lon;
        Math::cartesianToSpherical(vertices[i], lat, lon);
        cells[i].latitude = lat;
        cells[i].longitude = lon;

        // Initialize default attributes
        cells[i].elevation = 0.0f; // Sea level by default
        cells[i].plate_id = -1;
        cells[i].plate_velocity = Vector3(0, 0, 0);
        cells[i].temperature = 0.0f;
        cells[i].moisture = 0.0f;
        cells[i].precipitation = 0.0f;
        cells[i].wind_velocity = Vector3(0, 0, 0);
        cells[i].bedrock = RockType::BASALT;
        cells[i].soil = SoilType::NONE;
        cells[i].crustal_age = 0.0f;
        cells[i].crustal_thickness = 35.0f;
        cells[i].sediment_depth = 0.0f;
    }

    // Build neighbor adjacency list using unordered_sets for O(1) dedup
    std::vector<std::unordered_set<size_t>> neighborSets(vertices.size());

    for (const auto& tri : triangles) {
        for (int i = 0; i < 3; ++i) {
            size_t vA = tri.v[i];
            size_t vB = tri.v[(i + 1) % 3];
            neighborSets[vA].insert(vB);
            neighborSets[vB].insert(vA);
        }
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        cells[i].neighbors.assign(neighborSets[i].begin(), neighborSets[i].end());
    }
}

void GoldbergPolyhedron::generate(int subdivisionLevel) {
    buildBaseIcosahedron();
    
    for (int i = 0; i < subdivisionLevel; ++i) {
        subdivide();
    }
    
    extractCells();
}

} // namespace Ravis
