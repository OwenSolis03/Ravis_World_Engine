#pragma once

namespace Ravis {

struct SimulationParameters {
    float sea_level = 0.45f;
    float tectonic_speed = 0.05f;
    float orogenesis_factor = 1.0f; // Multiplier for mountain uplift
    float erosion_rate = 0.15f;     // General multiplier for droplet erosion
    float temp_offset = 0.0f;       // Shift global temp (Ice age vs Global warming)
};

} // namespace Ravis
