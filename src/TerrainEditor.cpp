#include "../include/TerrainEditor.h"
#include "../include/MathUtils.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace Ravis {

namespace {
// Keep user edits inside the range the renderers and climate expect.
constexpr float kMinElevation_m = -12000.0f;
constexpr float kMaxElevation_m = 10000.0f;

inline float geodesicDistanceRad(const Vector3& a, const Vector3& b) {
    double d = Math::dotProduct(a, b);
    d = std::max(-1.0, std::min(1.0, d));
    return static_cast<float>(std::acos(d));
}
} // namespace

void TerrainEditor::captureBase(const GoldbergPolyhedron& planet) {
    const auto& cells = planet.getCells();
    base_elevation.resize(cells.size());
    for (size_t i = 0; i < cells.size(); ++i)
        base_elevation[i] = cells[i].elevation;
    ops.clear();
    next_stroke_id = 0;
    current_stroke_id = 0;
}

size_t TerrainEditor::strokeCount() const {
    size_t count = 0;
    uint32_t last = UINT32_MAX;
    for (const auto& op : ops) {
        if (op.stroke_id != last) {
            ++count;
            last = op.stroke_id;
        }
    }
    return count;
}

// BFS out from the cell nearest to the dab center, through the neighbor graph,
// applying the cosine falloff to every cell within the geodesic radius. BFS
// (instead of a scan over all cells) keeps a dab O(cells touched).
bool TerrainEditor::addOpDelta(GoldbergPolyhedron& planet, const EditOp& op) {
    auto& cells = planet.getCells();
    if (cells.empty() || op.radius_rad <= 0.0f) return false;

    int start = pickCell(planet, op.center);
    if (start < 0) return false;

    const float sign = (op.type == EditOpType::RAISE) ? 1.0f : -1.0f;
    bool changed = false;

    std::vector<char> visited(cells.size(), 0);
    std::queue<size_t> frontier;
    frontier.push(static_cast<size_t>(start));
    visited[start] = 1;

    while (!frontier.empty()) {
        size_t i = frontier.front();
        frontier.pop();

        float d = geodesicDistanceRad(cells[i].position, op.center);
        if (d > op.radius_rad) continue; // outside the brush; don't expand

        // Δh(d) = amount * 0.5 * (1 + cos(pi * d / radius))
        float falloff = 0.5f * (1.0f + std::cos(static_cast<float>(Math::PI) * d / op.radius_rad));
        float delta = sign * op.amount_m * falloff;
        float next = cells[i].elevation + delta;
        next = std::max(kMinElevation_m, std::min(kMaxElevation_m, next));
        if (next != cells[i].elevation) {
            cells[i].elevation = next;
            changed = true;
        }

        for (size_t nid : cells[i].neighbors) {
            if (!visited[nid]) {
                visited[nid] = 1;
                frontier.push(nid);
            }
        }
    }
    return changed;
}

bool TerrainEditor::applyDab(GoldbergPolyhedron& planet, EditOpType type,
                             const Vector3& center, float radius_rad, float amount_m) {
    if (!hasBase() || planet.getCells().size() != base_elevation.size()) return false;

    EditOp op;
    op.type = type;
    op.center = Math::normalize(center);
    op.radius_rad = radius_rad;
    op.amount_m = amount_m;
    op.stroke_id = current_stroke_id;

    if (!addOpDelta(planet, op)) return false;
    ops.push_back(op);
    return true;
}

void TerrainEditor::recompose(GoldbergPolyhedron& planet) const {
    auto& cells = planet.getCells();
    if (cells.size() != base_elevation.size()) return;
    for (size_t i = 0; i < cells.size(); ++i)
        cells[i].elevation = base_elevation[i];
    for (const auto& op : ops)
        addOpDelta(planet, op);
}

bool TerrainEditor::undoLastStroke(GoldbergPolyhedron& planet) {
    if (ops.empty()) return false;
    uint32_t last = ops.back().stroke_id;
    while (!ops.empty() && ops.back().stroke_id == last)
        ops.pop_back();
    recompose(planet);
    return true;
}

void TerrainEditor::clearEdits(GoldbergPolyhedron& planet) {
    ops.clear();
    recompose(planet);
}

int TerrainEditor::pickCell(const GoldbergPolyhedron& planet, const Vector3& p) {
    const auto& cells = planet.getCells();
    if (cells.empty()) return -1;
    int best = 0;
    double bestDot = -2.0;
    for (size_t i = 0; i < cells.size(); ++i) {
        double d = Math::dotProduct(cells[i].position, p);
        if (d > bestDot) {
            bestDot = d;
            best = static_cast<int>(i);
        }
    }
    return best;
}

} // namespace Ravis
