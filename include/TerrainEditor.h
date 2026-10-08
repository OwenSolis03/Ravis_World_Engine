#pragma once

#include "GoldbergPolyhedron.h"
#include <cstdint>
#include <vector>

namespace Ravis {

enum class EditOpType { RAISE, LOWER };

// One brush dab, stored as an *operation* (center + radius + Δh), not as
// per-cell deltas. Operations are resolution-independent and replayable, so
// the same edit stack can be re-rasterized onto a finer mesh later.
struct EditOp {
    EditOpType type;
    Vector3 center;      // unit-sphere position of the dab
    float radius_rad;    // geodesic radius in radians
    float amount_m;      // peak |Δh| in meters (sign comes from type)
    uint32_t stroke_id;  // dabs of one mouse-drag share a stroke (undo unit)
};

// Authoritative user-edit layer on top of the generated terrain:
//   elevation = base_elevation (h_generated, immutable while editing) + Σ ops
// Falloff per dab: Δh(d) = amount * 0.5 * (1 + cos(pi * d / radius)), with d
// the geodesic distance acos(dot(p, center)) on the unit sphere.
class TerrainEditor {
public:
    // Snapshot the freshly generated world as the immutable base layer and
    // drop any previous edits. Call once after full generation completes.
    void captureBase(const GoldbergPolyhedron& planet);

    bool hasBase() const { return !base_elevation.empty(); }
    size_t opCount() const { return ops.size(); }
    size_t strokeCount() const;

    void beginStroke() { current_stroke_id = next_stroke_id++; }

    // Apply one dab incrementally to the cells and record it.
    // Returns true if any cell changed.
    bool applyDab(GoldbergPolyhedron& planet, EditOpType type,
                  const Vector3& center, float radius_rad, float amount_m);

    // Remove the dabs of the last stroke and recompose elevation = base + ops.
    // Returns true if a stroke was removed.
    bool undoLastStroke(GoldbergPolyhedron& planet);

    // Drop all edits and restore the generated terrain.
    void clearEdits(GoldbergPolyhedron& planet);

    // Nearest cell to a unit-sphere point (linear scan; fast enough even at
    // subdivision 8). Returns -1 if the planet has no cells.
    static int pickCell(const GoldbergPolyhedron& planet, const Vector3& p);

private:
    std::vector<float> base_elevation; // h_generated, immutable while editing
    std::vector<EditOp> ops;
    uint32_t next_stroke_id = 0;
    uint32_t current_stroke_id = 0;

    // elevation = base + Σ ops (canonical composition, used by undo/clear)
    void recompose(GoldbergPolyhedron& planet) const;

    // Add (or re-add) one op's delta onto the current elevation field.
    static bool addOpDelta(GoldbergPolyhedron& planet, const EditOp& op);
};

} // namespace Ravis
