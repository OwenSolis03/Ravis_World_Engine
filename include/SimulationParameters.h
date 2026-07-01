#pragma once

namespace Ravis {

struct SimulationParameters {
    // World seed (0 = random, >0 = deterministic)
    int seed = 0;

    // Planet
    int subdivision_level = 5;      // Icosahedron subdivision (5 = ~10K cells, 8 = ~655K)

    // Tectonics & Age
    float planet_age_Myr = 500.0f;  // Age of the planet in Millions of Years (Ma)
    int num_plates = 15;            // Number of tectonic plates
    float crust_fraction = 0.3f;    // Continental crust fraction (0.0 to 1.0)
    float avg_plate_speed_cm_yr = 5.0f; // Average plate speed in cm/year
    float thermal_subsidence_rate = 350.0f; // Half-space cooling constant
    float orogenesis_factor = 1.0f; // Multiplier for mountain uplift
    float primary_clustering = 0.3f;  // Pangea-like supercontinent clustering (0=dispersed, 1=single mass)
    float secondary_clustering = 0.2f; // America-like medium fragment clustering
    float crust_warping = 0.3f;     // Erratic landmass shape distortion
    bool use_bisector_distance = false; // Boundary Math toggle (true = Bisector, false = Cellular Propagation)

    // Geophysics
    int superswell_freq = 2;        // African-style mantle superswells (broad uplift ~1500km)
    int shallow_plume_freq = 10;    // Hawaii-style shallow magma plumes (island chains)
    int deep_plume_freq = 3;        // Iceland-style deep mantle plumes (large volcanic provinces)
    int old_mountain_freq = 5;      // Appalachian-style eroded ancient mountains
    int old_hill_freq = 8;          // Caledonian-style ancient eroded hills
    int small_uplift_freq = 15;     // Small stochastic terrain uplifts
    int upland_freq = 6;            // Volga-style broad gentle uplands

    // Climate
    float sea_level = 0.0f;         // Sea level in meters (0m default)
    float temp_offset = 0.0f;       // Global temperature shift (Ice age / Warming)
    int swe_iterations = 100;       // CUDA iterations for Atmospheric SWE solver
    int moisture_iterations = 20;   // Steps of moisture advection

    // Paleoclimate
    float lgm_temp_anomaly = 0.0f;  // Last Glacial Maximum temperature anomaly (°C, typically -5 to -10)
    float post_lgm_sea_rise = 0.0f; // Post-LGM sea level increase (meters, historically ~130m)

    // Erosion
    int num_drops = 200000;         // Hydraulic erosion droplet count
    float erosion_rate = 0.15f;     // Multiplier for droplet erosion

    // Backward-compat helpers (used internally, map old names to new)
    float land_ratio() const { return crust_fraction; }
    float tectonic_speed() const { return avg_plate_speed_cm_yr * 0.01f; }
    float effective_sea_level() const { return sea_level + post_lgm_sea_rise; }
    int tectonic_iterations() const { return static_cast<int>(planet_age_Myr / 5.0f); }
};

} // namespace Ravis
