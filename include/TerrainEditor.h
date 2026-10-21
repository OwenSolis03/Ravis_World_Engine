#pragma once

#include "GoldbergPolyhedron.h"
#include <cstdint>
#include <vector>

namespace Ravis {

enum class EditOpType { RAISE, LOWER };

struct EditOp {
    EditOpType type;
    Vector3 center;      // Posición en la esfera unitaria
    float radius_rad;    // Radio geodésico en radianes
    float amount_m;      // Pico |Δh| en metros
    uint32_t stroke_id;  // Identificador del trazo para Undo
};

class TerrainEditor {
public:
    void captureBase(const GoldbergPolyhedron& planet);

    bool hasBase() const { return !base_elevation.empty(); }
    size_t opCount() const { return ops.size(); }
    size_t strokeCount() const;

    void beginStroke() { current_stroke_id = next_stroke_id++; }

    bool applyDab(GoldbergPolyhedron& planet, EditOpType type,
                  const Vector3& center, float radius_rad, float amount_m);

    bool undoLastStroke(GoldbergPolyhedron& planet);
    void clearEdits(GoldbergPolyhedron& planet);

    // Restaura exactamente elevation = base_elevation + Σ ops y actualiza `locked`
    void recompose(GoldbergPolyhedron& planet) const;

    // Búsqueda rápida por ascenso de colina (hill-climbing) en el grafo de vecinos
    static int pickCell(const GoldbergPolyhedron& planet, const Vector3& p, int hint_cell = -1);

private:
    std::vector<float> base_elevation;
    std::vector<EditOp> ops;
    uint32_t next_stroke_id = 0;
    uint32_t current_stroke_id = 0;

    static bool addOpDelta(GoldbergPolyhedron& planet, const EditOp& op);
};

} // namespace Ravis