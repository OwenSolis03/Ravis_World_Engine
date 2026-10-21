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
    if (v1 > v2) std::swap(v1, v2);

    auto key = std::make_pair(v1, v2);
    auto it = midpointCache.find(key);
    if (it != midpointCache.end()) {
        return it->second;
    }

    const Vector3& p1 = vertices[v1];
    const Vector3& p2 = vertices[v2];
    Vector3 mid(
        (p1.x + p2.x) * 0.5,
        (p1.y + p2.y) * 0.5,
        (p1.z + p2.z) * 0.5
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

    // 5 caras alrededor del punto 0
    triangles.emplace_back(0, 11, 5);
    triangles.emplace_back(0, 5, 1);
    triangles.emplace_back(0, 1, 7);
    triangles.emplace_back(0, 7, 10);
    triangles.emplace_back(0, 10, 11);

    // 5 caras adyacentes
    triangles.emplace_back(1, 5, 9);
    triangles.emplace_back(5, 11, 4);
    triangles.emplace_back(11, 10, 2);
    triangles.emplace_back(10, 7, 6);
    triangles.emplace_back(7, 1, 8);

    // 5 caras alrededor del punto 3
    triangles.emplace_back(3, 9, 4);
    triangles.emplace_back(3, 4, 2);
    triangles.emplace_back(3, 2, 6);
    triangles.emplace_back(3, 6, 8);
    triangles.emplace_back(3, 8, 9);

    // 5 caras adyacentes
    triangles.emplace_back(4, 9, 5);
    triangles.emplace_back(2, 4, 11);
    triangles.emplace_back(6, 2, 10);
    triangles.emplace_back(8, 6, 7);
    triangles.emplace_back(9, 8, 1);
}

void GoldbergPolyhedron::subdivide() {
    std::vector<Triangle> newTriangles;
    newTriangles.reserve(triangles.size() * 4);
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
    triangles = std::move(newTriangles);
}

// Relaja los vértices de grado 6 hacia el baricentro de sus celdas Voronoi duales
// manteniendo fijos los 12 vértices pentagonales originales para homogeneizar áreas.
void GoldbergPolyhedron::relaxMesh(int passes) {
    if (passes <= 0 || vertices.size() <= 12) return;

    const size_t n = vertices.size();
    std::vector<std::vector<size_t>> adj(n);
    for (size_t i = 0; i < n; ++i) adj[i].reserve(6);

    for (const auto& tri : triangles) {
        for (int k = 0; k < 3; ++k) {
            size_t u = tri.v[k];
            size_t v = tri.v[(k + 1) % 3];
            if (std::find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) adj[u].push_back(v);
            if (std::find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) adj[v].push_back(u);
        }
    }

    std::vector<Vector3> nextVertices(n);
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < n; ++i) {
            if (i < 12 || adj[i].empty()) {
                nextVertices[i] = vertices[i];
                continue;
            }
            Vector3 avg(0.0, 0.0, 0.0);
            for (size_t nid : adj[i]) {
                avg.x += vertices[nid].x;
                avg.y += vertices[nid].y;
                avg.z += vertices[nid].z;
            }
            double inv = 1.0 / static_cast<double>(adj[i].size());
            Vector3 target = Math::normalize(Vector3(avg.x * inv, avg.y * inv, avg.z * inv));
            // Relajación amortiguada (0.35) proyectada a la esfera unitaria
            nextVertices[i] = Math::normalize(Vector3(
                vertices[i].x * 0.65 + target.x * 0.35,
                vertices[i].y * 0.65 + target.y * 0.35,
                vertices[i].z * 0.65 + target.z * 0.35
            ));
        }
        vertices = nextVertices;
    }
}

void GoldbergPolyhedron::extractCells() {
    const size_t n = vertices.size();
    cells.clear();
    cells.resize(n);

    // Construir lista de adyacencia sin duplicados (máximo 6 vecinos por celda)
    std::vector<std::vector<size_t>> adj(n);
    for (size_t i = 0; i < n; ++i) adj[i].reserve(6);

    for (const auto& tri : triangles) {
        for (int k = 0; k < 3; ++k) {
            size_t vA = tri.v[k];
            size_t vB = tri.v[(k + 1) % 3];
            if (std::find(adj[vA].begin(), adj[vA].end(), vB) == adj[vA].end())
                adj[vA].push_back(vB);
            if (std::find(adj[vB].begin(), adj[vB].end(), vA) == adj[vB].end())
                adj[vB].push_back(vA);
        }
    }

    rowMajorNeighbors.assign(n * 6, -1);
    transposedNeighbors.assign(n * 6, -1);
    neighborCounts.assign(n, 0);

    for (size_t i = 0; i < n; ++i) {
        cells[i].id = i;
        cells[i].position = vertices[i];

        double lat, lon;
        Math::cartesianToSpherical(vertices[i], lat, lon);
        cells[i].latitude = lat;
        cells[i].longitude = lon;

        // Estado inicial limpio
        cells[i].elevation = 0.0f;
        cells[i].is_oceanic = false;
        cells[i].plate_id = -1;
        cells[i].plate_velocity = Vector3(0, 0, 0);
        cells[i].locked = false;
        cells[i].temperature = 0.0f;
        cells[i].temp_seasonality = 0.0f;
        cells[i].moisture = 0.0f;
        cells[i].precipitation = 0.0f;
        cells[i].wind_velocity = Vector3(0, 0, 0);
        cells[i].ocean_current = Vector3(0, 0, 0);
        cells[i].ocean_temp_anomaly = 0.0f;
        cells[i].bedrock = RockType::BASALT;
        cells[i].soil = SoilType::NONE;
        cells[i].biome = BiomeType::OCEAN;
        cells[i].crustal_age = 0.0f;
        cells[i].crustal_thickness = 35.0f;
        cells[i].sediment_depth = 0.0f;
        cells[i].is_lake = false;
        cells[i].water_surface_elev = 0.0f;
        cells[i].river_flow = 0.0f;
        cells[i].downstream_id = i;

        // Ordenar vecinos en sentido antihorario en el plano tangente local (Este, Norte)
        Vector3 east, north;
        Math::localTangentBasis(vertices[i], east, north);
        auto& nbrs = adj[i];
        std::sort(nbrs.begin(), nbrs.end(), [&](size_t a, size_t b) {
            Vector3 da(vertices[a].x - vertices[i].x, vertices[a].y - vertices[i].y, vertices[a].z - vertices[i].z);
            Vector3 db(vertices[b].x - vertices[i].x, vertices[b].y - vertices[i].y, vertices[b].z - vertices[i].z);
            double angA = std::atan2(Math::dotProduct(da, north), Math::dotProduct(da, east));
            double angB = std::atan2(Math::dotProduct(db, north), Math::dotProduct(db, east));
            return angA < angB;
        });

        cells[i].neighbors = std::move(nbrs);

        int count = static_cast<int>(std::min<size_t>(6, cells[i].neighbors.size()));
        neighborCounts[i] = count;
        for (int k = 0; k < count; ++k) {
            int nid = static_cast<int>(cells[i].neighbors[k]);
            rowMajorNeighbors[i * 6 + k] = nid;
            transposedNeighbors[k * n + i] = nid;
        }
    }
}

void GoldbergPolyhedron::generate(int subdivisionLevel) {
    buildBaseIcosahedron();

    for (int i = 0; i < subdivisionLevel; ++i) {
        subdivide();
    }

    // 2 pasadas de relajación esférica suavizan la variación de área pentágono-hexágono
    relaxMesh(2);
    extractCells();
}

} // namespace Ravis