#pragma once

namespace Ravis {

struct SimulationParameters {
    // World seed (0 = random, >0 = deterministic)
    int seed = 0;

    // Planet (8 = ~655K cells, ~30km resolution; fast and detailed default)
    int subdivision_level = 8;

    // Tectonics & Age (Calibrated to Modern Earth / Phanerozoic Wilson Cycle)
    float planet_age_Myr = 550.0f;          // Active Phanerozoic tectonic window (Ma)
    int num_plates = 14;                    // 7 major + 7 minor plates (Earth configuration)
    float crust_fraction = 0.35f;           // Yields ~29% emergent land above sea level (Earth = 29.2%)
    float avg_plate_speed_cm_yr = 5.0f;     // Earth average plate speed (~5 cm/yr)
    float thermal_subsidence_rate = 220.0f; // Stein & Stein plate cooling -> mean ocean depth ~ -3,800m
    float orogenesis_factor = 0.90f;        // Mean land elevation ~840m, max peaks ~8,848m
    float primary_clustering = 0.25f;       // Moderate Eurasia/Africa-style grouping (avoids extreme Pangea)
    float secondary_clustering = 0.25f;     // Americas-style secondary continental grouping
    float crust_warping = 0.35f;            // Natural coastline irregularity
    bool use_bisector_distance = false;     // Cellular propagation for organic fault belts

    // Geophysics (Earth hotspot & cratonic province frequencies)
    int superswell_freq = 2;        // African & Pacific superswells
    int shallow_plume_freq = 12;    // Hawaii / Canary / Galapagos island chains
    int deep_plume_freq = 3;        // Iceland / Yellowstone / Afar deep plumes
    int old_mountain_freq = 4;      // Appalachians / Urals / Great Dividing Range
    int old_hill_freq = 6;          // Caledonian / Brazilian shield hills
    int small_uplift_freq = 10;     // Regional massifs
    int upland_freq = 5;            // Central Siberian / Volga uplands

    // Climate (Earth-like energy & moisture transport)
    float sea_level = 0.0f;         // Base sea level (0m)
    float temp_offset = -0.02f;     // Global mean surface temperature ~14.8 °C
    int swe_iterations = 150;       // Well-converged geostrophic/monsoonal winds
    int moisture_iterations = 32;   // ~3,000 km inland moisture transport reach

    // Paleoclimate (Holocene interglacial default)
    float lgm_temp_anomaly = 0.0f;  // 0°C = Modern Holocene (-6°C = Last Glacial Maximum)
    float post_lgm_sea_rise = 0.0f; // 0m = Modern sea level

    // Erosion
    int num_drops = 220000;
    float erosion_rate = 0.18f;

    // Helpers
    float land_ratio() const { return crust_fraction; }
    float tectonic_speed() const { return avg_plate_speed_cm_yr * 0.01f; }
    float effective_sea_level() const { return sea_level + post_lgm_sea_rise; }
    int tectonic_iterations() const { return static_cast<int>(planet_age_Myr / 5.0f); }
};

} // namespace Ravis