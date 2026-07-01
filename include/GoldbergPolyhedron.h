#pragma once

#include "PlanetData.h"
#include <vector>
#include <unordered_map>

namespace Ravis {

// Custom hash for std::pair<size_t, size_t> to enable unordered_map usage
struct PairHash {
    size_t operator()(const std::pair<size_t,size_t>& p) const {
        return std::hash<size_t>()(p.first) ^
               (std::hash<size_t>()(p.second) << 32);
    }
};

class GoldbergPolyhedron {
public:
    GoldbergPolyhedron();

    // Generates the Goldberg Polyhedron based on the subdivision level of an Icosahedron.
    // Each vertex of the resulting icosphere becomes a Cell (hexagon or pentagon) in our simulation.
    // subdivisionLevel 0: 12 cells (all pentagons)
    // subdivisionLevel 1: 42 cells
    // subdivisionLevel 2: 162 cells
    void generate(int subdivisionLevel);

    const std::vector<Cell>& getCells() const { return cells; }
    std::vector<Cell>& getCells() { return cells; }

    struct Triangle {
        size_t v[3];
        Triangle(size_t v1, size_t v2, size_t v3) {
            v[0] = v1; v[1] = v2; v[2] = v3;
        }
    };

    const std::vector<Vector3>& getVertices() const { return vertices; }
    const std::vector<Triangle>& getTriangles() const { return triangles; }

private:
    std::vector<Cell> cells;

    std::vector<Vector3> vertices;
    std::vector<Triangle> triangles;

    // Helper to add a vertex, normalizes it to sit on the unit sphere, and returns its index
    size_t addVertex(const Vector3& v);

    // Caches midpoint indices to avoid duplicating vertices during subdivision
    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> midpointCache;
    size_t getMidpointIndex(size_t v1, size_t v2);

    void buildBaseIcosahedron();
    void subdivide();
    void extractCells();
};

} // namespace Ravis
