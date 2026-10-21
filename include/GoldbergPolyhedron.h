#pragma once

#include "PlanetData.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Ravis {

// Hash simétrico para std::pair<size_t, size_t>
struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        return std::hash<size_t>()(p.first) ^
               (std::hash<size_t>()(p.second) * 0x9e3779b97f4a7c15ULL + (std::hash<size_t>()(p.first) << 6));
    }
};

class GoldbergPolyhedron {
public:
    GoldbergPolyhedron();

    // Genera el poliedro de Goldberg subdividiendo un icosaedro base,
    // relaja la malla sobre la esfera y ordena topológicamente los vecinos.
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

    // Buffers planos precalculados para CUDA:
    // rowMajorNeighbors: [num_cells * 6] (compatible con kernels actuales i * 6 + n)
    // transposedNeighbors: [6 * num_cells] (para acceso coalescente n * num_cells + i)
    const std::vector<int>& getRowMajorNeighbors() const { return rowMajorNeighbors; }
    const std::vector<int>& getTransposedNeighbors() const { return transposedNeighbors; }
    const std::vector<int>& getNeighborCounts() const { return neighborCounts; }

private:
    std::vector<Cell> cells;
    std::vector<Vector3> vertices;
    std::vector<Triangle> triangles;

    std::vector<int> rowMajorNeighbors;
    std::vector<int> transposedNeighbors;
    std::vector<int> neighborCounts;

    size_t addVertex(const Vector3& v);

    std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> midpointCache;
    size_t getMidpointIndex(size_t v1, size_t v2);

    void buildBaseIcosahedron();
    void subdivide();
    void relaxMesh(int passes);
    void extractCells();
};

} // namespace Ravis