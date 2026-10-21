#include "../include/AtmosphereSimulator.h"
#include "../include/ErosionSimulator.h"
#include "../include/GoldbergPolyhedron.h"
#include "../include/MapExporter.h"
#include "../include/MathUtils.h"
#include "../include/HydrologySimulator.h"
#include "../include/OceanSimulator.h"
#include "../include/PedologySimulator.h"
#include "../include/SimulationParameters.h"
#include "../include/TectonicSimulator.h"
#include "../include/TerrainEditor.h"
#include "../include/VegetationSimulator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ImGui & GLFW
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <memory>

using namespace Ravis;

// Background Simulation State
std::atomic<bool> is_simulating(false);
std::atomic<bool> new_map_ready(false);
std::mutex pixel_mutex;
std::thread sim_thread;

// 2D map. Only the active layer is kept as pixels: at subdivision 10 the map is
// 16384x8192 (400 MB per RGB layer), so a layer is re-rendered from the planet
// through the pixel->cell lookup when the user switches to it.
std::vector<unsigned char> map_pixels; // RGB of layer map_pixels_layer
int map_pixels_layer = -1;
int map_width = 0, map_height = 0;     // size of map_pixels and map_cell_lookup
int map_dirty_y0 = 0, map_dirty_y1 = 0; // rows of map_pixels the texture still lacks
int max_texture_size = 16384;          // GL_MAX_TEXTURE_SIZE, queried at startup
int active_map_index = 0; // 0:Biome, 1:Height, 2:Lithology, 3:Pedology, 4:Hydrology, 5:Temperature, 6:Moisture
std::atomic<int> requested_map_layer(0); // active_map_index, readable from the sim thread

// Map width in pixels for a planet with `cell_count` cells: 2^(level+4), which
// gives every hexagon of subdivision `level` at least 2x2 pixels (measured for
// levels 8-10). Kept within [2048, GL_MAX_TEXTURE_SIZE]; the height is width / 2.
static int mapWidthForCells(size_t cell_count) {
    int level = 0;
    while (level < 12 && 10ull * (1ull << (2 * level)) + 2 < cell_count) ++level;
    int width = 1 << std::min(level + 4, 14);
    return std::max(2048, std::min(width, max_texture_size));
}

std::shared_ptr<GoldbergPolyhedron> active_planet = nullptr;

// UI-editable parameters (global so the input callbacks can trigger recomputes)
SimulationParameters params;
std::atomic<int> last_world_seed(0); // seed actually used by the last generation

// Terrain editing state
TerrainEditor terrain_editor;
bool edit_mode = false;
int brush_mode = 0;              // 0 = Raise, 1 = Lower
float brush_radius_km = 400.0f;  // geodesic radius, Earth-like radius assumed
float brush_strength_m = 800.0f; // peak delta per dab, meters
bool is_painting = false;
int last_painted_cell = -1;
bool stroke_made_changes = false;
GLFWwindow* g_window = nullptr;
const float kPlanetRadiusKm = 6371.0f; // brush km -> radians conversion

// 3D rendering data
std::vector<float> globe_vertices; // [x,y,z, r,g,b]
std::vector<unsigned int> globe_indices;
// Orbit camera: it always looks at the point (camera_lat, camera_lon) of the
// planet from camera_distance planet radii, with north up on screen.
float camera_distance = 3.0f;
float camera_lon = 0.0f; // radians, east positive
float camera_lat = 0.0f; // radians, north positive
bool is_dragging = false;
double last_mouse_x, last_mouse_y;
bool show_3d = false; // the 2D map is the default view
bool update_3d_geometry = false;
bool update_3d_colors = false;
const float kCameraFovY = 45.0f * 3.1415926535f / 180.0f;
const float kCameraMinDistance = 1.15f;
const float kCameraMaxDistance = 8.0f;

// 2D map editing state
PixelCellMap map_cell_lookup; // pixel -> cell of the current planet
bool is_painting_2d = false;
bool map_preview_dirty = false; // elevation changed under the 2D map
// Latitude band (radians) touched by brush dabs since the last map refresh,
// so only those rows are re-rendered while painting.
double preview_lat_min = 1e9, preview_lat_max = -1e9;

// 2D map view: zoom factor and the texture UV at the center of the view.
float map_zoom = 1.0f;
double map_center_u = 0.5, map_center_v = 0.5;
const float kMaxMapZoom = 64.0f;

// Rotation taking model space (planet axis = +Z) to view space, as three
// columns: the looked-at point lands on +Z (towards the camera), north on +Y
// and east on +X.
static void cameraRotation(Vector3& col_x, Vector3& col_y, Vector3& col_z) {
    double sl = std::sin(camera_lat), cl = std::cos(camera_lat);
    double so = std::sin(camera_lon), co = std::cos(camera_lon);
    col_x = Vector3(-so, -sl * co, cl * co);
    col_y = Vector3(co, -sl * so, cl * so);
    col_z = Vector3(0.0, cl, sl);
}

// Unproject the mouse onto the globe. Inverse of the render transform
// (view = T(0,0,-dist) * R * model, R from cameraRotation), intersecting the
// view-space ray with the unit sphere centered at (0,0,-dist).
// Returns the model-space point on the unit sphere.
static bool pickSpherePoint(GLFWwindow* window, double mx, double my, Vector3& out_model) {
    int ww = 0, wh = 0;
    glfwGetWindowSize(window, &ww, &wh);
    if (ww <= 0 || wh <= 0) return false;

    // Mouse -> view-space ray (same 45 deg vertical FOV as the projection)
    float ndc_x = 2.0f * static_cast<float>(mx) / ww - 1.0f;
    float ndc_y = 1.0f - 2.0f * static_cast<float>(my) / wh;
    float aspect = static_cast<float>(ww) / static_cast<float>(wh);
    float tan_half = tanf(kCameraFovY * 0.5f);
    Vector3 dir = Math::normalize(Vector3(ndc_x * tan_half * aspect, ndc_y * tan_half, -1.0));

    // |t*dir - c|^2 = 1 with c = (0,0,-dist):  t^2 - 2t(dir.c) + |c|^2 - 1 = 0
    Vector3 c(0.0, 0.0, -camera_distance);
    double b = Math::dotProduct(dir, c);
    double disc = b * b - (Math::dotProduct(c, c) - 1.0);
    if (disc < 0.0) return false; // ray misses the globe
    double t = b - std::sqrt(disc); // near intersection
    if (t < 0.0) return false;

    // Back to model space: q = view point relative to sphere center,
    // model = R^T * q
    Vector3 q(dir.x * t - c.x, dir.y * t - c.y, dir.z * t - c.z);
    Vector3 col_x, col_y, col_z;
    cameraRotation(col_x, col_y, col_z);
    out_model = Math::normalize(Vector3(Math::dotProduct(col_x, q),
                                        Math::dotProduct(col_y, q),
                                        Math::dotProduct(col_z, q)));
    return true;
}

// Equirectangular map pixel <-> unit-sphere point.
static Vector3 mapUVToSphere(double u, double v) {
    double lat = Math::PI * 0.5 - v * Math::PI;
    double lon = u * 2.0 * Math::PI - Math::PI;
    return Math::sphericalToCartesian(lat, lon);
}

static void sphereToMapUV(const Vector3& p, double& u, double& v) {
    double lat, lon;
    Math::cartesianToSpherical(p, lat, lon);
    u = (lon + Math::PI) / (2.0 * Math::PI);
    v = (Math::PI * 0.5 - lat) / Math::PI;
}

// RGB pixels of one map layer, for `height` rows of `width` pixels described
// by `lookup` (the whole map or a band of rows of it).
static std::vector<unsigned char> renderMapLayer(const GoldbergPolyhedron& planet, int layer,
                                                 int width, int height, float sea,
                                                 const PixelCellMap& lookup) {
    switch (layer) {
    case 0: return MapExporter::getBiomePixels(planet, width, height, sea, lookup);
    case 1: return MapExporter::getHeightPixels(planet, width, height, sea, lookup);
    case 2: return MapExporter::getLithologyPixels(planet, width, height, sea, lookup);
    case 3: return MapExporter::getPedologyPixels(planet, width, height, sea, lookup);
    case 4: return MapExporter::getHydrologyPixels(planet, width, height, sea, lookup);
    case 5: return MapExporter::getTemperaturePixels(planet, width, height, sea, lookup);
    default: return MapExporter::getMoisturePixels(planet, width, height, sea, lookup);
    }
}

// Render the active 2D map layer from the planet and publish it as the active
// world. The pixel->cell lookup depends only on the mesh, so a recompute after
// a brush edit (same planet object) reuses it. Runs on the simulation thread;
// the UI thread does not touch the lookup while is_simulating is set.
static void publishMaps(std::shared_ptr<GoldbergPolyhedron> planet,
                        const SimulationParameters& p) {
    float eff_sea = p.effective_sea_level();
    int width = mapWidthForCells(planet->getCells().size());
    int height = width / 2;

    bool reuse;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        reuse = planet == active_planet && map_width == width && map_height == height &&
                map_cell_lookup.size() == static_cast<size_t>(width) * height;
    }
    PixelCellMap fresh;
    if (!reuse) fresh = MapExporter::buildPixelToCellMap(*planet, width, height);
    const PixelCellMap& lookup = reuse ? map_cell_lookup : fresh;
    int layer = requested_map_layer.load();
    auto pixels = renderMapLayer(*planet, layer, width, height, eff_sea, lookup);

    std::lock_guard<std::mutex> lock(pixel_mutex);
    if (!reuse) map_cell_lookup = std::move(fresh);
    map_pixels = std::move(pixels);
    map_pixels_layer = layer;
    map_width = width;
    map_height = height;
    map_dirty_y0 = 0;
    map_dirty_y1 = height;
    active_planet = planet;
    new_map_ready = true;
    update_3d_geometry = true;
    update_3d_colors = true;
}

// Wall-clock time per pipeline stage
class StageTimer {
public:
    void lap(const char* name) {
        auto now = std::chrono::steady_clock::now();
        stages.emplace_back(name, std::chrono::duration<double>(now - last).count());
        last = now;
    }
    void report(const char* title) const {
        std::ostringstream o;
        double total = 0.0;
        for (const auto& s : stages) total += s.second;
        o << std::fixed << std::setprecision(2) << "-- " << title << " (" << total << " s) --\n";
        for (const auto& s : stages)
            o << "  " << std::left << std::setw(22) << s.first << std::right
              << std::setw(7) << s.second << " s\n";
        o << "\n";
        std::cout << o.str() << std::flush;
        std::ofstream f("worldgen.log", std::ios::app);
        if (f) f << o.str();
    }
private:
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, double>> stages;
};

// Re-derive climate -> ocean -> hydrology -> biomes -> soils from the (edited)
// elevation field, without regenerating the mesh or re-running tectonics and erosion.
void recomputeDerivedFields(SimulationParameters p) {
    std::shared_ptr<GoldbergPolyhedron> planet;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        planet = active_planet;
        if (planet) {
            terrain_editor.recompose(*planet);
        }
    }
    if (!planet) {
        is_simulating = false;
        return;
    }

    p.seed = last_world_seed.load(); // keep noise fields identical to generation

    std::cout << "Recomputing climate after terrain edit..." << std::endl;
    StageTimer timer;
    AtmosphereSimulator atmosphere(*planet);
    atmosphere.calculatePrimaryClimate(p);
    timer.lap("primary climate");

    OceanSimulator ocean(*planet);
    ocean.simulateCurrents(p);

    atmosphere.calculateFullClimate(p.moisture_iterations, p);
    timer.lap("winds + moisture");

    HydrologySimulator hydrology(*planet);
    hydrology.simulate(p);
    timer.lap("hydrology");

    atmosphere.assignBiomes(p);

    PedologySimulator pedology(*planet);
    pedology.classifyBedrock(p);
    pedology.generateSoils(p);

    VegetationSimulator vegetation(*planet);
    vegetation.simulate(p);

    // Re-asignar biomas finales tras la regulación térmica e hídrica de la vegetación
    atmosphere.assignBiomes(p);
    timer.lap("biomes + soils");

    publishMaps(planet, p);
    timer.lap("2D maps");
    timer.report("Recompute timing");
    std::cout << "Recompute complete." << std::endl;
    is_simulating = false;
}

static void startRecompute() {
    if (is_simulating) return;
    is_simulating = true;
    if (sim_thread.joinable()) sim_thread.join();
    sim_thread = std::thread(recomputeDerivedFields, params);
}

// Apply one brush dab centered on the unit-sphere point p.
static void paintAtPoint(const Vector3& p, int cell) {
    if (is_simulating) return;
    std::shared_ptr<GoldbergPolyhedron> planet;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        planet = active_planet;
    }
    if (!planet || !terrain_editor.hasBase()) return;

    if (cell < 0) cell = TerrainEditor::pickCell(*planet, p, last_painted_cell);
    if (cell < 0 || cell == last_painted_cell) return;
    last_painted_cell = cell;

    float radius_rad = brush_radius_km / kPlanetRadiusKm;
    EditOpType type = (brush_mode == 0) ? EditOpType::RAISE : EditOpType::LOWER;
    if (terrain_editor.applyDab(*planet, type, p, radius_rad, brush_strength_m)) {
        stroke_made_changes = true;
        update_3d_colors = true;
        map_preview_dirty = true;
        double lat, lon;
        Math::cartesianToSpherical(p, lat, lon);
        preview_lat_min = std::min(preview_lat_min, lat - radius_rad);
        preview_lat_max = std::max(preview_lat_max, lat + radius_rad);
    }
}

// Brush dab under the cursor on the 3D globe.
static void paintAtCursor(double mx, double my) {
    if (!show_3d) return;
    Vector3 p;
    if (!pickSpherePoint(g_window, mx, my, p)) return;
    paintAtPoint(p, -1);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (action == GLFW_PRESS && ImGui::GetIO().WantCaptureMouse) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            if (edit_mode && show_3d && !is_simulating && active_planet) {
                is_painting = true;
                stroke_made_changes = false;
                last_painted_cell = -1;
                terrain_editor.beginStroke();
                paintAtCursor(last_mouse_x, last_mouse_y);
            } else {
                is_dragging = true;
            }
        } else if (action == GLFW_RELEASE) {
            is_dragging = false;
            if (is_painting) {
                is_painting = false;
                if (stroke_made_changes) startRecompute();
            }
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) is_dragging = true;
        else if (action == GLFW_RELEASE) is_dragging = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (is_painting) {
        paintAtCursor(xpos, ypos);
        last_mouse_x = xpos;
        last_mouse_y = ypos;
        return;
    }
    if (is_dragging) {
        int ww = 0, wh = 0;
        glfwGetWindowSize(window, &ww, &wh);
        float rad_per_px = 2.0f * tanf(kCameraFovY * 0.5f) *
                           (camera_distance - 1.0f) / static_cast<float>(std::max(wh, 1));
        float lon_scale = 1.0f / std::max(cosf(camera_lat), 0.2f);
        camera_lon -= static_cast<float>(xpos - last_mouse_x) * rad_per_px * lon_scale;
        camera_lat += static_cast<float>(ypos - last_mouse_y) * rad_per_px;

        const float pi = 3.1415926535f;
        const float max_lat = pi * 0.5f - 0.01f;
        camera_lat = std::max(-max_lat, std::min(max_lat, camera_lat));
        if (camera_lon > pi) camera_lon -= 2.0f * pi;
        if (camera_lon < -pi) camera_lon += 2.0f * pi;
    }
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    float height = (camera_distance - 1.0f) * powf(0.85f, static_cast<float>(yoffset));
    camera_distance = std::max(kCameraMinDistance, std::min(kCameraMaxDistance, 1.0f + height));
}

static const char *biomeName(BiomeType b) {
    switch (b) {
    case BiomeType::OCEAN: return "Ocean";
    case BiomeType::ICE: return "Ice";
    case BiomeType::TUNDRA: return "Tundra";
    case BiomeType::BOREAL_FOREST: return "BorealForest";
    case BiomeType::TEMPERATE_FOREST: return "TemperateForest";
    case BiomeType::GRASSLAND: return "Grassland";
    case BiomeType::STEPPE: return "Steppe";
    case BiomeType::RAINFOREST: return "Rainforest";
    case BiomeType::TEMPERATE_RAINFOREST: return "TemperateRainforest";
    case BiomeType::TROPICAL_DRY_FOREST: return "TropicalDryForest";
    case BiomeType::MEDITERRANEAN: return "Mediterranean";
    case BiomeType::DESERT: return "Desert";
    case BiomeType::SAVANNA: return "Savanna";
    case BiomeType::THORN_SCRUB: return "ThornScrub";
    }
    return "?";
}

static void logWorldStats(const GoldbergPolyhedron &planet,
                          const SimulationParameters &params) {
    const auto &cells = planet.getCells();
    if (cells.empty())
        return;

    const double sea = params.effective_sea_level();
    const size_t n = cells.size();

    size_t land = 0, ocean = 0, landAbove1k = 0, landAbove3k = 0;
    double landElevSum = 0.0, oceanDepthSum = 0.0;
    float landMax = -1e30f, oceanMin = 1e30f;
    std::vector<float> landElev;
    landElev.reserve(n);

    size_t crustOceanic = 0, crustContinental = 0;
    double crustOceanicElevSum = 0.0, crustContinentalElevSum = 0.0;

    double tSum = 0.0, pSumLand = 0.0, mSumLand = 0.0;
    float tMin = 1e30f, tMax = -1e30f;
    size_t lakeCells = 0, riverCells = 0;
    std::map<std::string, size_t> biomeCount;
    std::array<size_t, 5> rockCount{};
    std::array<size_t, 4> soilCount{};

    for (const auto &c : cells) {
        tSum += c.temperature;
        tMin = std::min(tMin, c.temperature);
        tMax = std::max(tMax, c.temperature);

        if (c.is_oceanic) {
            crustOceanic++;
            crustOceanicElevSum += c.elevation;
        } else {
            crustContinental++;
            crustContinentalElevSum += c.elevation;
        }
        if (c.is_lake)
            lakeCells++;
        if (c.river_flow > 0.001f)
            riverCells++;

        int ri = static_cast<int>(c.bedrock);
        if (ri >= 0 && ri < 5)
            rockCount[ri]++;

        if (c.elevation > sea) {
            land++;
            int si = static_cast<int>(c.soil);
            if (si >= 0 && si < 4)
                soilCount[si]++;
            float e = static_cast<float>(c.elevation - sea);
            landElevSum += e;
            landElev.push_back(e);
            if (e > landMax)
                landMax = e;
            if (e > 1000.0f)
                landAbove1k++;
            if (e > 3000.0f)
                landAbove3k++;
            pSumLand += c.precipitation;
            mSumLand += c.moisture;
            biomeCount[biomeName(c.biome)]++;
        } else {
            ocean++;
            oceanDepthSum += (c.elevation - sea);
            if (c.elevation < oceanMin)
                oceanMin = c.elevation;
        }
    }

    std::sort(landElev.begin(), landElev.end());
    auto q = [&](double f) -> float {
        if (landElev.empty())
            return 0.0f;
        return landElev[static_cast<size_t>(f * (landElev.size() - 1))];
    };
    auto pctN = [&](size_t x) { return n ? 100.0 * x / n : 0.0; };
    auto pctL = [&](size_t x) { return land ? 100.0 * x / land : 0.0; };

    std::ostringstream o;
    std::time_t tt = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &tt);
#else
    localtime_r(&tt, &tmv);
#endif

    o << std::fixed << std::setprecision(1);
    o << "================================================================\n";
    o << "[" << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S")
      << "]  seed=" << params.seed << "\n";
    o << "-- Parameters --\n";
    o << "  subdivision_level       " << params.subdivision_level << "\n";
    o << "  num_plates              " << params.num_plates << "\n";
    o << "  crust_fraction          " << params.crust_fraction << "\n";
    o << "  planet_age_Myr          " << params.planet_age_Myr << "\n";
    o << "  avg_plate_speed_cm_yr   " << params.avg_plate_speed_cm_yr << "\n";
    o << "  thermal_subsidence_rate " << params.thermal_subsidence_rate << "\n";
    o << "  orogenesis_factor       " << params.orogenesis_factor << "\n";
    o << "  primary_clustering      " << params.primary_clustering << "\n";
    o << "  secondary_clustering    " << params.secondary_clustering << "\n";
    o << "  crust_warping           " << params.crust_warping << "\n";
    o << "  use_bisector_distance   "
      << (params.use_bisector_distance ? "true" : "false") << "\n";
    o << "  superswell_freq         " << params.superswell_freq << "\n";
    o << "  shallow_plume_freq      " << params.shallow_plume_freq << "\n";
    o << "  deep_plume_freq         " << params.deep_plume_freq << "\n";
    o << "  old_mountain_freq       " << params.old_mountain_freq << "\n";
    o << "  old_hill_freq           " << params.old_hill_freq << "\n";
    o << "  small_uplift_freq       " << params.small_uplift_freq << "\n";
    o << "  upland_freq             " << params.upland_freq << "\n";
    o << "  sea_level               " << params.sea_level << "\n";
    o << "  temp_offset             " << params.temp_offset << "\n";
    o << "  swe_iterations          " << params.swe_iterations << "\n";
    o << "  moisture_iterations     " << params.moisture_iterations << "\n";
    o << "  num_drops               " << params.num_drops << "\n";
    o << "  erosion_rate            " << params.erosion_rate << "\n";
    o << "  lgm_temp_anomaly        " << params.lgm_temp_anomaly << "\n";
    o << "  post_lgm_sea_rise       " << params.post_lgm_sea_rise << "\n";
    o << "  effective_sea_level     " << sea << " m\n";

    o << "-- Geography (" << n << " cells) --\n";
    o << "  land / ocean            " << land << " (" << pctN(land) << "%)  /  "
      << ocean << " (" << pctN(ocean) << "%)\n";
    o << "  continental elev  mean  " << (land ? landElevSum / land : 0.0)
      << " m   median " << q(0.5) << "   p90 " << q(0.9) << "   max " << landMax
      << " m\n";
    o << "    land > 1000 m         " << landAbove1k << " (" << pctL(landAbove1k)
      << "% of land)\n";
    o << "    land > 3000 m         " << landAbove3k << " (" << pctL(landAbove3k)
      << "% of land)\n";
    o << "  oceanic depth     mean  " << (ocean ? oceanDepthSum / ocean : 0.0)
      << " m   deepest " << (oceanMin - sea) << " m\n";
    o << "  crust flag oceanic      " << crustOceanic << "   mean elev "
      << (crustOceanic ? crustOceanicElevSum / crustOceanic : 0.0) << " m\n";
    o << "  crust flag continental  " << crustContinental << "   mean elev "
      << (crustContinental ? crustContinentalElevSum / crustContinental : 0.0)
      << " m\n";

    o << "-- Climate --\n";
    o << std::setprecision(3);
    o << "  temperature norm  mean  " << (n ? tSum / n : 0.0) << "   [" << tMin
      << " .. " << tMax << "]\n";
    o << std::setprecision(1);
    o << "    approx deg C   mean   " << (n ? (tSum / n) * 55.0 - 15.0 : 0.0)
      << "   [" << (tMin * 55.0 - 15.0) << " .. " << (tMax * 55.0 - 15.0)
      << "]\n";
    o << std::setprecision(3);
    o << "  precipitation norm mean " << (land ? pSumLand / land : 0.0)
      << "  (land)\n";
    o << std::setprecision(0);
    o << "    approx mm/yr   mean   " << (land ? (pSumLand / land) * 4000.0 : 0.0)
      << "\n";
    o << std::setprecision(3);
    o << "  moisture norm      mean " << (land ? mSumLand / land : 0.0)
      << "  (land)\n";
    o << std::setprecision(1);
    o << "  lake cells " << lakeCells << "   river cells " << riverCells << "\n";

    o << "-- Biomes (land) --\n";
    for (const auto &kv : biomeCount)
        o << "  " << std::left << std::setw(22) << kv.first << std::right
          << kv.second << " (" << pctL(kv.second) << "%)\n";

    static const char *rockNames[5] = {"Basalt", "Granite", "Sandstone",
                                       "ShaleLimestone", "Metamorphic"};
    o << "-- Bedrock (all cells) --\n";
    for (int i = 0; i < 5; ++i)
        o << "  " << std::left << std::setw(22) << rockNames[i] << std::right
          << rockCount[i] << " (" << pctN(rockCount[i]) << "%)\n";

    static const char *soilNames[4] = {"None", "Sand", "Clay", "Loam"};
    o << "-- Soil (land) --\n";
    for (int i = 0; i < 4; ++i)
        o << "  " << std::left << std::setw(22) << soilNames[i] << std::right
          << soilCount[i] << " (" << pctL(soilCount[i]) << "%)\n";
    o << "\n";

    std::cout << o.str() << std::flush;
    std::ofstream f("worldgen.log", std::ios::app);
    if (f)
        f << o.str();
    else
        std::cerr << "WARN: could not open worldgen.log for writing\n";
}

void runSimulation(SimulationParameters params) {
    if (params.seed == 0) {
        params.seed = static_cast<int>(
            std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
    }

    std::cout << "Starting simulation (seed: " << params.seed << ")..."
              << std::endl;
    auto new_planet = std::make_shared<GoldbergPolyhedron>();
    StageTimer timer;

    new_planet->generate(params.subdivision_level);
    timer.lap("mesh");

    TectonicSimulator tectonics(*new_planet);
    tectonics.generatePlates(params.num_plates, params);
    tectonics.simulate(params.tectonic_iterations(), params);
    timer.lap("tectonics");

    AtmosphereSimulator atmosphere(*new_planet);
    atmosphere.calculatePrimaryClimate(params);
    timer.lap("primary climate");

    ErosionSimulator erosion(*new_planet);
    int scaled_drops = static_cast<int>(params.num_drops * (params.planet_age_Myr / 500.0f));
    erosion.simulateErosion(scaled_drops, params);
    timer.lap("erosion");

    OceanSimulator ocean(*new_planet);
    ocean.simulateCurrents(params);

    atmosphere.calculateFullClimate(params.moisture_iterations, params);
    timer.lap("winds + moisture");

    HydrologySimulator hydrology(*new_planet);
    hydrology.simulate(params);
    timer.lap("hydrology");

    // Re-assign biomes after hydrology (riparian moisture changes biomes)
    atmosphere.assignBiomes(params);

    PedologySimulator pedology(*new_planet);
    pedology.classifyBedrock(params);
    pedology.generateSoils(params);

    VegetationSimulator vegetation(*new_planet);
    vegetation.simulate(params);

    // Re-asignar biomas finales tras la regulación térmica e hídrica de la vegetación
    atmosphere.assignBiomes(params);
    timer.lap("biomes + soils");

    logWorldStats(*new_planet, params);

    last_world_seed = params.seed;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        terrain_editor.captureBase(*new_planet);
    }

    publishMaps(new_planet, params);
    timer.lap("2D maps");
    timer.report("Generation timing");

    std::cout << "Simulation complete! (seed: " << params.seed << ")"
              << std::endl;
    is_simulating = false;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--headless") {
        if (argc > 2) params.subdivision_level = std::atoi(argv[2]);
        runSimulation(params);
        return 0;
    }

    if (!glfwInit())
        return -1;

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow *window =
        glfwCreateWindow(1280, 720, "Ravis World Engine", NULL, NULL);
    if (!window)
        return -1;

    // The window is always maximized: restoring it (title-bar double click,
    // Win+Down) maximizes it again. Minimizing still works.
    glfwMaximizeWindow(window);
    glfwSetWindowMaximizeCallback(window, [](GLFWwindow* w, int maximized) {
        if (!maximized && !glfwGetWindowAttrib(w, GLFW_ICONIFIED))
            glfwMaximizeWindow(w);
    });

    g_window = window;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    GLint gl_max_texture = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &gl_max_texture);
    if (gl_max_texture > 0) max_texture_size = std::min(16384, static_cast<int>(gl_max_texture));

    // glGenerateMipmap is GL 3.0, absent from the Windows GL 1.1 headers.
    typedef void (APIENTRY *GenerateMipmapFn)(GLenum);
    auto glGenerateMipmapPtr =
        reinterpret_cast<GenerateMipmapFn>(glfwGetProcAddress("glGenerateMipmap"));

    GLuint map_texture = 0;
    int map_texture_w = 0, map_texture_h = 0; // size the texture was allocated with

    glGenTextures(1, &map_texture);
    glBindTexture(GL_TEXTURE_2D, map_texture);
    // Mipmaps keep the zoomed-out view of a 16384-wide map from aliasing;
    // nearest magnification keeps hexagon edges sharp when zoomed in.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    glGenerateMipmapPtr ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI Panel
        ImGui::Begin("World Engine Controller");

        // --- World Seed ---
        ImGui::Text("World Generation");
        ImGui::Separator();

        ImGui::InputInt("Seed", &params.seed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = Random seed each generation. Any other value = deterministic.");
        ImGui::SameLine();
        if (ImGui::SmallButton("Randomize")) {
            params.seed = static_cast<int>(
                std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
        }

        ImGui::SliderInt("Subdivision", &params.subdivision_level, 3, 10);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Mesh resolution. (5 = 10K, 6 = 40K, 7 = 163K, 8 = 655K, 9 = 2.6M, 10 = 10.5M cells)\nHigher = slower: level 10 takes about a minute and a half.");
        }

        ImGui::Spacing();

        // --- Tectonics ---
        ImGui::Text("Tectonic Parameters");
        ImGui::Separator();

        ImGui::SliderInt("Plates", &params.num_plates, 5, 30);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Number of tectonic plates. More plates = more fragmented crust.");

        ImGui::SliderFloat("Crust Fraction", &params.crust_fraction, 0.1f, 0.9f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Continental crust fraction. Higher = more landmass.");

        ImGui::SliderFloat("Planet Age (Myr)", &params.planet_age_Myr, 10.0f, 4500.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Age of the planet in Millions of Years. Controls tectonic steps, erosion, and subsidence.");

        ImGui::SliderFloat("Plate Speed (cm/yr)", &params.avg_plate_speed_cm_yr, 0.5f, 20.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Average plate speed in cm/year. Earth avg ~5 cm/yr.");

        ImGui::SliderFloat("Orogenesis", &params.orogenesis_factor, 0.1f, 3.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Mountain uplift multiplier. High = massive Himalayan ranges.");

        ImGui::SliderFloat("Primary Clustering", &params.primary_clustering, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Pangea-like supercontinent grouping. 1.0 = single landmass.");

        ImGui::SliderFloat("Secondary Clustering", &params.secondary_clustering, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("America-like medium fragment clustering in the opposite hemisphere.");

        ImGui::SliderFloat("Crust Warping", &params.crust_warping, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Erratic landmass distortion. Higher = more irregular continental edges.");

        ImGui::Checkbox("Use Bisector Boundary Math", &params.use_bisector_distance);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Toggle between mathematical seed bisector distance and exact cellular propagation distance for fault borders.");

        ImGui::Spacing();

        // --- Geophysics ---
        ImGui::Text("Geophysics");
        ImGui::Separator();

        ImGui::SliderInt("Superswells", &params.superswell_freq, 0, 10);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("African-style mantle superswells. Broad ~1500km doming of the crust.");

        ImGui::SliderInt("Shallow Plumes", &params.shallow_plume_freq, 0, 30);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hawaii-style shallow magma plumes. Small volcanic island chains.");

        ImGui::SliderInt("Deep Plumes", &params.deep_plume_freq, 0, 15);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Iceland-style deep mantle plumes. Large volcanic provinces.");

        ImGui::SliderInt("Old Mountains", &params.old_mountain_freq, 0, 20);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Appalachian-style ancient eroded mountains (800-1500m).");

        ImGui::SliderInt("Old Hills", &params.old_hill_freq, 0, 30);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Caledonian-style ancient eroded hills (400-800m).");

        ImGui::SliderInt("Small Uplifts", &params.small_uplift_freq, 0, 50);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Small stochastic terrain uplifts (200-500m).");

        ImGui::SliderInt("Uplands", &params.upland_freq, 0, 20);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Volga-style broad gentle uplands (300-600m).");

        ImGui::Spacing();

        // --- Climate & Erosion ---
        ImGui::Text("Climate & Erosion");
        ImGui::Separator();

        ImGui::SliderFloat("Sea Level (m)", &params.sea_level, -500.0f, 500.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Base sea level in meters. 0m = default.");

        ImGui::SliderFloat("Temp Offset", &params.temp_offset, -0.5f, 0.5f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Global temperature shift. Negative = Ice Age.");

        ImGui::SliderFloat("Subsidence Rate", &params.thermal_subsidence_rate, 100.0f, 600.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Half-space cooling rate for oceanic crust. Higher = sinks faster.");

        ImGui::SliderInt("CUDA Iterations", &params.swe_iterations, 10, 5000);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("SWE solver iterations on GPU. Higher = more stable winds.");

        ImGui::SliderInt("Moisture Steps", &params.moisture_iterations, 5, 50);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Steps of moisture advection. More = smoother moisture.");

        ImGui::SliderInt("Erosion Drops", &params.num_drops, 50000, 500000);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Number of hydraulic erosion droplets. More = smoother terrain.");

        ImGui::SliderFloat("Erosion Rate", &params.erosion_rate, 0.01f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How aggressively water carves through rock.");

        ImGui::Spacing();

        // --- Paleoclimate ---
        ImGui::Text("Paleoclimate");
        ImGui::Separator();

        ImGui::SliderFloat("LGM Temp Anomaly (C)", &params.lgm_temp_anomaly, -10.0f, 0.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Last Glacial Maximum temperature drop. Earth LGM was approx -5 to -8 C.");

        ImGui::SliderFloat("Post-LGM Sea Rise (m)", &params.post_lgm_sea_rise, 0.0f, 130.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sea level rise after ice age. Earth: ~130m since LGM.");

        float eff = params.effective_sea_level();
        ImGui::Text("Effective Sea Level: %.1f m", eff);

        ImGui::Spacing();

        // --- Terrain Editing ---
        ImGui::Text("Terrain Editing");
        ImGui::Separator();

        if (ImGui::Checkbox("Edit Mode", &edit_mode) && edit_mode)
            show_3d = false;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paint on the 2D map with Left Click + Drag.\n"
                              "On release, climate, rivers and biomes are recomputed\n"
                              "around your edit. The 3D globe can be painted too\n"
                              "(rotate it with Right Click + Drag).");

        if (edit_mode) {
            const char *brush_items[] = {"Raise", "Lower"};
            ImGui::Combo("Brush", &brush_mode, brush_items, IM_ARRAYSIZE(brush_items));

            ImGui::SliderFloat("Brush Radius (km)", &brush_radius_km, 50.0f, 2000.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Geodesic brush radius (Earth-sized planet assumed).");

            ImGui::SliderFloat("Brush Strength (m)", &brush_strength_m, 50.0f, 3000.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Peak elevation change per dab, with cosine falloff.");

            if (!is_simulating) {
                std::lock_guard<std::mutex> lock(pixel_mutex);
                ImGui::Text("Strokes: %zu (dabs: %zu)", terrain_editor.strokeCount(),
                            terrain_editor.opCount());
            }

            if (!is_simulating && terrain_editor.strokeCount() > 0) {
                std::shared_ptr<GoldbergPolyhedron> edit_planet;
                {
                    std::lock_guard<std::mutex> lock(pixel_mutex);
                    edit_planet = active_planet;
                }
                if (edit_planet) {
                    if (ImGui::Button("Undo Stroke")) {
                        if (terrain_editor.undoLastStroke(*edit_planet)) {
                            update_3d_colors = true;
                            startRecompute();
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Clear All Edits")) {
                        terrain_editor.clearEdits(*edit_planet);
                        update_3d_colors = true;
                        startRecompute();
                    }
                }
            }
            if (!terrain_editor.hasBase())
                ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1),
                                   "Generate a world first.");
        }

        ImGui::Separator();

        if (is_simulating) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1),
                               "Simulating in background... Please wait.");
        } else {
            if (ImGui::Button("Generate World", ImVec2(200, 40))) {
                is_simulating = true;
                if (sim_thread.joinable())
                    sim_thread.join();
                sim_thread = std::thread(runSimulation, params);
            }
        }

        ImGui::End();

        ImGui::Begin("World Map Output", nullptr,
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        const char *items[] = {"Biome Map", "Height Map", "Lithology (Rock)",
                               "Pedology (Soil)", "Hydrology", "Temperature", "Moisture"};
        bool map_changed = ImGui::Combo("Map Layer", &active_map_index, items,
                                        IM_ARRAYSIZE(items));

        requested_map_layer = active_map_index;
        if (map_changed && !is_simulating)
            update_3d_colors = true; // the 2D map re-renders itself below

        ImGui::Separator();
        ImGui::Checkbox("Show 3D Globe", &show_3d);
        if (show_3d) {
            if (edit_mode) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Left Click + Drag to Paint");
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Right Click + Drag to Rotate");
            } else {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Left Click + Drag to Rotate");
            }
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Scroll to Zoom");
            if (ImGui::Button("Reset Camera")) {
                camera_lat = 0.0f;
                camera_lon = 0.0f;
                camera_distance = 3.0f;
            }
        }

        if (!show_3d && map_texture != 0) {
            std::shared_ptr<GoldbergPolyhedron> map_planet;
            {
                std::lock_guard<std::mutex> lock(pixel_mutex);
                if (!is_simulating) map_planet = active_planet;
            }
            bool lookup_ok = map_planet && map_width > 0 &&
                             map_cell_lookup.size() == static_cast<size_t>(map_width) * map_height;

            // Re-render the active layer when the user switched layers, or the
            // rows a brush stroke touched (raw terrain change while painting;
            // the full recompute on release refreshes everything).
            if (lookup_ok && (map_pixels_layer != active_map_index || map_preview_dirty)) {
                int y0 = 0, y1 = map_height;
                if (map_pixels_layer == active_map_index && preview_lat_min <= preview_lat_max) {
                    auto latToRow = [&](double lat) {
                        return (Math::PI * 0.5 - lat) / Math::PI * (map_height - 1);
                    };
                    y0 = std::max(0, static_cast<int>(std::floor(latToRow(preview_lat_max))) - 1);
                    y1 = std::min(map_height, static_cast<int>(std::ceil(latToRow(preview_lat_min))) + 2);
                }
                if (y1 > y0) {
                    float eff_sea = params.effective_sea_level();
                    bool full = (y0 == 0 && y1 == map_height);
                    std::vector<unsigned char> pixels;
                    if (full) {
                        pixels = renderMapLayer(*map_planet, active_map_index, map_width, map_height,
                                                eff_sea, map_cell_lookup);
                    } else {
                        PixelCellMap band(map_cell_lookup.begin() + static_cast<size_t>(y0) * map_width,
                                          map_cell_lookup.begin() + static_cast<size_t>(y1) * map_width);
                        pixels = renderMapLayer(*map_planet, active_map_index, map_width, y1 - y0,
                                                eff_sea, band);
                    }
                    std::lock_guard<std::mutex> lock(pixel_mutex);
                    if (full) {
                        map_pixels = std::move(pixels);
                    } else {
                        std::copy(pixels.begin(), pixels.end(),
                                  map_pixels.begin() + static_cast<size_t>(y0) * map_width * 3);
                    }
                    map_pixels_layer = active_map_index;
                    if (new_map_ready) { // merge with an upload still pending
                        y0 = std::min(y0, map_dirty_y0);
                        y1 = std::max(y1, map_dirty_y1);
                    }
                    map_dirty_y0 = y0;
                    map_dirty_y1 = y1;
                    new_map_ready = true;
                }
            }
            map_preview_dirty = false;
            preview_lat_min = 1e9;
            preview_lat_max = -1e9;

            if (edit_mode)
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Left Click + Drag on the map to Paint");
            ImGui::TextDisabled(edit_mode ? "Scroll: zoom   Right/Middle drag: pan"
                                          : "Scroll: zoom   Drag: pan");
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset View")) {
                map_zoom = 1.0f;
                map_center_u = 0.5;
                map_center_v = 0.5;
            }
            if (map_width > 0) {
                ImGui::SameLine();
                ImGui::TextDisabled("  %dx%d px  zoom x%.1f", map_width, map_height, map_zoom);
            }

            // Largest 2:1 map that fits, leaving a line below for the cell readout.
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float fit_h = avail.y - ImGui::GetTextLineHeightWithSpacing();
            float map_w = std::max(256.0f, std::min(avail.x, fit_h * 2.0f));
            ImVec2 map_size(map_w, map_w * 0.5f);
            ImVec2 map_min = ImGui::GetCursorScreenPos();

            // Visible window of the texture, in UV. The view stays inside the map.
            double span = 1.0 / map_zoom;
            map_center_u = std::max(span * 0.5, std::min(1.0 - span * 0.5, map_center_u));
            map_center_v = std::max(span * 0.5, std::min(1.0 - span * 0.5, map_center_v));
            double view_u0 = map_center_u - span * 0.5, view_v0 = map_center_v - span * 0.5;

            ImGui::Image((void *)(intptr_t)map_texture, map_size,
                         ImVec2((float)view_u0, (float)view_v0),
                         ImVec2((float)(view_u0 + span), (float)(view_v0 + span)));
            // An invisible button over the image takes the mouse, so dragging on
            // the map paints / pans instead of moving the window.
            ImGui::SetCursorScreenPos(map_min);
            ImGui::InvisibleButton("map_canvas", map_size,
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                                   ImGuiButtonFlags_MouseButtonMiddle);
            bool map_hovered = ImGui::IsItemHovered();
            bool map_active = ImGui::IsItemActive();

            auto screenToUV = [&](const ImVec2& s, double& u, double& v) {
                u = view_u0 + std::max(0.0, std::min(1.0, (double)(s.x - map_min.x) / map_size.x)) * span;
                v = view_v0 + std::max(0.0, std::min(1.0, (double)(s.y - map_min.y) / map_size.y)) * span;
            };
            auto uvToScreen = [&](double u, double v) {
                return ImVec2(map_min.x + (float)((u - view_u0) / span) * map_size.x,
                              map_min.y + (float)((v - view_v0) / span) * map_size.y);
            };

            // Zoom towards the cursor: the map point under it stays put.
            ImGuiIO& map_io = ImGui::GetIO();
            if (map_hovered && map_io.MouseWheel != 0.0f) {
                double u, v;
                screenToUV(map_io.MousePos, u, v);
                float new_zoom = std::max(1.0f, std::min(kMaxMapZoom, map_zoom * powf(1.25f, map_io.MouseWheel)));
                double fx = (map_io.MousePos.x - map_min.x) / map_size.x;
                double fy = (map_io.MousePos.y - map_min.y) / map_size.y;
                double new_span = 1.0 / new_zoom;
                map_center_u = u - (fx - 0.5) * new_span;
                map_center_v = v - (fy - 0.5) * new_span;
                map_zoom = new_zoom;
            }
            bool panning = map_active && (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                                          ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                                          (!edit_mode && ImGui::IsMouseDown(ImGuiMouseButton_Left)));
            if (panning) {
                map_center_u -= map_io.MouseDelta.x / map_size.x * span;
                map_center_v -= map_io.MouseDelta.y / map_size.y * span;
            }

            if (map_hovered && lookup_ok) {
                double u, v;
                screenToUV(ImGui::GetMousePos(), u, v);
                int px = static_cast<int>(u * (map_width - 1) + 0.5);
                int py = static_cast<int>(v * (map_height - 1) + 0.5);
                uint32_t cell_id = map_cell_lookup[static_cast<size_t>(py) * map_width + px];
                Vector3 p = mapUVToSphere(u, v);

                const Cell& hovered = map_planet->getCells()[cell_id];
                ImGui::Text("Cell %u   Lat %.2f  Lon %.2f   Elevation %.0f m", cell_id,
                            hovered.latitude * 180.0 / Math::PI,
                            hovered.longitude * 180.0 / Math::PI,
                            hovered.elevation - params.effective_sea_level());

                if (edit_mode && terrain_editor.hasBase()) {
                    // Brush outline: the geodesic circle of the brush, which is
                    // not a circle on an equirectangular map.
                    double radius_rad = brush_radius_km / kPlanetRadiusKm;
                    Vector3 ref = std::abs(p.z) < 0.9 ? Vector3(0, 0, 1) : Vector3(1, 0, 0);
                    Vector3 a = Math::normalize(Math::crossProduct(p, ref));
                    Vector3 b = Math::crossProduct(p, a);
                    ImDrawList* draw = ImGui::GetWindowDrawList();
                    draw->PushClipRect(map_min, ImVec2(map_min.x + map_size.x, map_min.y + map_size.y), true);
                    const int kSegments = 64;
                    ImVec2 prev;
                    double prev_u = 0.0;
                    for (int s = 0; s <= kSegments; ++s) {
                        double ang = 2.0 * Math::PI * s / kSegments;
                        double ca = std::cos(ang) * std::sin(radius_rad);
                        double sa = std::sin(ang) * std::sin(radius_rad);
                        double cr = std::cos(radius_rad);
                        Vector3 q(cr * p.x + ca * a.x + sa * b.x,
                                  cr * p.y + ca * a.y + sa * b.y,
                                  cr * p.z + ca * a.z + sa * b.z);
                        double qu, qv;
                        sphereToMapUV(q, qu, qv);
                        ImVec2 cur = uvToScreen(qu, qv);
                        // Skip the segment that jumps across the +-180 seam.
                        if (s > 0 && std::abs(qu - prev_u) < 0.5)
                            draw->AddLine(prev, cur, IM_COL32(255, 255, 255, 220), 1.5f);
                        prev = cur;
                        prev_u = qu;
                    }
                    draw->PopClipRect();

                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        is_painting_2d = true;
                        stroke_made_changes = false;
                        last_painted_cell = -1;
                        terrain_editor.beginStroke();
                    }
                    if (is_painting_2d && ImGui::IsMouseDown(ImGuiMouseButton_Left))
                        paintAtPoint(p, static_cast<int>(cell_id));
                }
            }

            if (is_painting_2d && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                is_painting_2d = false;
                if (stroke_made_changes) startRecompute();
            }
        }
        map_preview_dirty = false;
        ImGui::End();

        // Legends & Symbology Panel (synchronized with PlanetData.h biomeRGB and SoilType)
        ImGui::Begin("Legends & Symbology");
        if (active_map_index == 0) {
            ImGui::Text("Biome Colors");
            ImGui::Separator();
            const BiomeType legend_biomes[] = {
                BiomeType::OCEAN, BiomeType::ICE, BiomeType::TUNDRA, BiomeType::BOREAL_FOREST,
                BiomeType::TEMPERATE_FOREST, BiomeType::TEMPERATE_RAINFOREST, BiomeType::STEPPE,
                BiomeType::GRASSLAND, BiomeType::MEDITERRANEAN, BiomeType::DESERT,
                BiomeType::THORN_SCRUB, BiomeType::SAVANNA, BiomeType::TROPICAL_DRY_FOREST,
                BiomeType::RAINFOREST
            };
            const char* legend_names[] = {
                "Ocean", "Ice", "Tundra", "Boreal Forest",
                "Temperate Forest", "Temperate Rainforest", "Steppe",
                "Grassland", "Mediterranean", "Desert",
                "Thorn Scrub", "Savanna", "Tropical Dry Forest",
                "Tropical Rainforest"
            };
            for (int i = 0; i < 14; i++) {
                float r = 0.1f, g = 0.2f, b = 0.6f;
                if (legend_biomes[i] != BiomeType::OCEAN) {
                    biomeRGB(legend_biomes[i], r, g, b);
                }
                ImGui::PushID(i);
                ImGui::ColorButton("##b", ImVec4(r, g, b, 1.0f), ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker);
                ImGui::PopID();
                ImGui::SameLine();
                ImGui::Text("%s", legend_names[i]);
            }
        } else if (active_map_index == 1) {
            ImGui::Text("Elevation Scale");
            ImGui::Separator();
            ImGui::Text("White (Ice): Temperature < 0.2");
            ImGui::Text("Dark Brown (Peaks): 3000m - 10000m");
            ImGui::Text("Yellow-Green (Hills): 1000m - 3000m");
            ImGui::Text("Green (Plains): 0m - 1000m");
            ImGui::Text("Light Blue (Shallows): Sea Level");
            ImGui::Text("Dark Blue (Trenches): < -11000m");
        } else if (active_map_index == 2) {
            ImGui::Text("Lithology (Rock Type)");
            ImGui::Separator();
            ImGui::ColorButton("##1", ImVec4(30/255.f, 30/255.f, 30/255.f, 1)); ImGui::SameLine(); ImGui::Text("Basalt (Oceanic / Volcanic)");
            ImGui::ColorButton("##2", ImVec4(120/255.f, 120/255.f, 120/255.f, 1)); ImGui::SameLine(); ImGui::Text("Granite (Continental Craton)");
            ImGui::ColorButton("##3", ImVec4(210/255.f, 180/255.f, 140/255.f, 1)); ImGui::SameLine(); ImGui::Text("Sandstone (Clastic Sedimentary)");
            ImGui::ColorButton("##4", ImVec4(160/255.f, 160/255.f, 150/255.f, 1)); ImGui::SameLine(); ImGui::Text("Shale / Limestone (Carbonate / Basin)");
            ImGui::ColorButton("##5", ImVec4(200/255.f, 200/255.f, 255/255.f, 1)); ImGui::SameLine(); ImGui::Text("Metamorphic (Orogenic Core)");
        } else if (active_map_index == 3) {
            ImGui::Text("Pedology (Soil Type)");
            ImGui::Separator();
            ImGui::ColorButton("##1", ImVec4(100/255.f, 100/255.f, 100/255.f, 1)); ImGui::SameLine(); ImGui::Text("None (Ice / Bare High Rock)");
            ImGui::ColorButton("##2", ImVec4(240/255.f, 230/255.f, 140/255.f, 1)); ImGui::SameLine(); ImGui::Text("Sand (Aridisol / Coastal Dunes)");
            ImGui::ColorButton("##3", ImVec4(180/255.f, 100/255.f, 50/255.f, 1)); ImGui::SameLine(); ImGui::Text("Clay (Oxisol / Tropical Leached)");
            ImGui::ColorButton("##4", ImVec4(80/255.f, 40/255.f, 20/255.f, 1)); ImGui::SameLine(); ImGui::Text("Loam (Fertile Forest / Grassland)");
        } else if (active_map_index == 4) {
            ImGui::Text("Hydrology");
            ImGui::Separator();
            ImGui::ColorButton("##1", ImVec4(15/255.f, 15/255.f, 60/255.f, 1)); ImGui::SameLine(); ImGui::Text("Ocean");
            ImGui::ColorButton("##2", ImVec4(0.0f, 200/255.f, 220/255.f, 1)); ImGui::SameLine(); ImGui::Text("Lake / Inland Sea");
            ImGui::ColorButton("##3", ImVec4(20/255.f, 50/255.f, 240/255.f, 1)); ImGui::SameLine(); ImGui::Text("River (High Flow)");
            ImGui::ColorButton("##4", ImVec4(51/255.f, 51/255.f, 51/255.f, 1)); ImGui::SameLine(); ImGui::Text("Dry Land");
        } else if (active_map_index == 5) {
            ImGui::Text("Temperature");
            ImGui::Separator();
            ImGui::ColorButton("##1", ImVec4(1, 0, 0, 1)); ImGui::SameLine(); ImGui::Text("Hot (Equatorial)");
            ImGui::ColorButton("##2", ImVec4(0.5f, 0, 0.5f, 1)); ImGui::SameLine(); ImGui::Text("Moderate (Temperate)");
            ImGui::ColorButton("##3", ImVec4(0, 0, 1, 1)); ImGui::SameLine(); ImGui::Text("Cold (Polar)");
        } else if (active_map_index == 6) {
            ImGui::Text("Moisture / Precipitation");
            ImGui::Separator();
            ImGui::ColorButton("##1", ImVec4(0, 0.35f, 1, 1)); ImGui::SameLine(); ImGui::Text("Wet (Rainforest / Orographic)");
            ImGui::ColorButton("##2", ImVec4(0, 0.18f, 0.5f, 1)); ImGui::SameLine(); ImGui::Text("Moderate");
            ImGui::ColorButton("##3", ImVec4(0, 0, 0, 1)); ImGui::SameLine(); ImGui::Text("Dry (Desert)");
        }
        ImGui::End();

        // Update texture if new data arrived
        // Upload the map rows that changed: the whole map after a generation or
        // a layer switch, only the brushed band while painting.
        if (new_map_ready) {
            std::lock_guard<std::mutex> lock(pixel_mutex);
            if (!map_pixels.empty() && map_dirty_y1 > map_dirty_y0) {
                glBindTexture(GL_TEXTURE_2D, map_texture);
                bool resized = map_texture_w != map_width || map_texture_h != map_height;
                if (resized || (map_dirty_y0 == 0 && map_dirty_y1 == map_height)) {
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, map_width, map_height, 0, GL_RGB,
                                 GL_UNSIGNED_BYTE, map_pixels.data());
                    map_texture_w = map_width;
                    map_texture_h = map_height;
                } else {
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, map_dirty_y0, map_width,
                                    map_dirty_y1 - map_dirty_y0, GL_RGB, GL_UNSIGNED_BYTE,
                                    map_pixels.data() + static_cast<size_t>(map_dirty_y0) * map_width * 3);
                }
                if (glGenerateMipmapPtr) glGenerateMipmapPtr(GL_TEXTURE_2D);
            }
            map_dirty_y0 = map_dirty_y1 = 0;
            new_map_ready = false;
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update 3D Geometry if needed
        std::shared_ptr<GoldbergPolyhedron> current_planet;
        {
            std::lock_guard<std::mutex> lock(pixel_mutex);
            current_planet = active_planet;
        }

        // The globe is only built when shown: at subdivision 10 its arrays are
        // ~500 MB. The update flags stay set until then.
        if (current_planet && show_3d) {
            if (update_3d_geometry) {
                const auto& verts = current_planet->getVertices();
                const auto& tris = current_planet->getTriangles();
                globe_vertices.resize(verts.size() * 6);
                for (size_t i = 0; i < verts.size(); ++i) {
                    globe_vertices[i * 6 + 0] = static_cast<float>(verts[i].x);
                    globe_vertices[i * 6 + 1] = static_cast<float>(verts[i].y);
                    globe_vertices[i * 6 + 2] = static_cast<float>(verts[i].z);
                    globe_vertices[i * 6 + 3] = 1.0f;
                    globe_vertices[i * 6 + 4] = 1.0f;
                    globe_vertices[i * 6 + 5] = 1.0f;
                }
                globe_indices.resize(tris.size() * 3);
                for (size_t i = 0; i < tris.size(); ++i) {
                    globe_indices[i * 3 + 0] = static_cast<unsigned int>(tris[i].v[0]);
                    globe_indices[i * 3 + 1] = static_cast<unsigned int>(tris[i].v[1]);
                    globe_indices[i * 3 + 2] = static_cast<unsigned int>(tris[i].v[2]);
                }
                update_3d_geometry = false;
            }

            if (update_3d_colors && !globe_vertices.empty()) {
                const auto& cells = current_planet->getCells();
                const auto& verts = current_planet->getVertices();
                const float sea_level = params.effective_sea_level();

                // Calcular umbral de ríos (top 5% de caudal) para dibujarlos en el Globo 3D
                float river_threshold_3d = 0.0f;
                {
                    std::vector<float> flows_3d;
                    flows_3d.reserve(cells.size() / 4);
                    for (const auto& c : cells) {
                        if (c.elevation > sea_level && c.river_flow > 0.0f && !c.is_lake) {
                            flows_3d.push_back(c.river_flow);
                        }
                    }
                    if (!flows_3d.empty()) {
                        size_t kth = static_cast<size_t>(flows_3d.size() * 0.975);
                        std::nth_element(flows_3d.begin(), flows_3d.begin() + kth, flows_3d.end());
                        river_threshold_3d = flows_3d[kth];
                    }
                }

                // 1. Precalcular una elevación suavizada solo para el desplazamiento 3D
                // (evita que una sola celda alta forme una aguja puntiaguda)
                std::vector<float> smooth_elev(cells.size());
                for (size_t i = 0; i < cells.size(); ++i) {
                    float sum = cells[i].elevation * 2.0f;
                    float weight = 2.0f;
                    for (size_t nid : cells[i].neighbors) {
                        sum += cells[nid].elevation;
                        weight += 1.0f;
                    }
                    smooth_elev[i] = sum / weight;
                }

                // Dirección de luz solar en espacio del modelo (iluminación oblicua natural)
                const Vector3 light_dir = Math::normalize(Vector3(0.55, -0.45, 0.70));

                for (size_t i = 0; i < cells.size(); ++i) {
                    float r = 1.0f, g = 1.0f, b = 1.0f;
                    const auto& cell = cells[i];
                    float elev = cell.elevation;

                    // 2. Desplazamiento radial suave con compresión hiperbólica (tanh):
                    // - El océano permanece en una esfera perfecta (scale = 1.0f).
                    // - Las colinas y cordilleras tienen relieve visible, pero las cumbres
                    //   extremas (>4000m) se comprimen suavemente hasta un tope de +1.4% del radio.
                    float scale = 1.0f;
                    if (smooth_elev[i] > sea_level) {
                        float h_above = smooth_elev[i] - sea_level;
                        scale = 1.0f + 0.014f * std::tanh(h_above / 3800.0f);
                    }
                    globe_vertices[i * 6 + 0] = static_cast<float>(verts[i].x) * scale;
                    globe_vertices[i * 6 + 1] = static_cast<float>(verts[i].y) * scale;
                    globe_vertices[i * 6 + 2] = static_cast<float>(verts[i].z) * scale;

                    float max_slope = 0.0f;
                    for (size_t nid : cell.neighbors) {
                        float diff = fabsf(cells[nid].elevation - elev);
                        if (diff > max_slope) max_slope = diff;
                    }
                    bool is_steep = max_slope > 1000.0f;

                    if (active_map_index == 1) { // Height
                        if (elev <= sea_level) {
                            float depth = std::max(0.0f, std::min(1.0f, (elev + 11000.0f) / (sea_level + 11000.0f)));
                            r = 0.0f;
                            g = (100 * depth) / 255.0f;
                            b = (50 + 150 * depth) / 255.0f;
                            if (cell.temperature < 0.2f) {
                                r = 220/255.f; g = 240/255.f; b = 255/255.f;
                            }
                        } else {
                            float land = (elev - sea_level);
                            if (land < 1000.0f) {
                                float t = land / 1000.0f;
                                r = (34 * (1 - t) + 173 * t) / 255.0f;
                                g = (139 * (1 - t) + 255 * t) / 255.0f;
                                b = (34 * (1 - t) + 47 * t) / 255.0f;
                            } else if (land < 3000.0f) {
                                float t = (land - 1000.0f) / 2000.0f;
                                r = (173 * (1 - t) + 139 * t) / 255.0f;
                                g = (255 * (1 - t) + 69 * t) / 255.0f;
                                b = (47 * (1 - t) + 19 * t) / 255.0f;
                            } else {
                                float t = std::min(1.0f, (land - 3000.0f) / 5000.0f);
                                r = (139 * (1 - t) + 60 * t) / 255.0f;
                                g = (69 * (1 - t) + 30 * t) / 255.0f;
                                b = (19 * (1 - t) + 10 * t) / 255.0f;
                            }

                            if (cell.temperature < 0.2f && !is_steep) {
                                r = 1.0f; g = 1.0f; b = 1.0f;
                            }

                            // Superponer lagos y ríos en la vista de alturas 3D
                            if (cell.is_lake) {
                                r = 0.0f; g = 190/255.f; b = 240/255.f;
                            } else if (river_threshold_3d > 0.0f && cell.river_flow >= river_threshold_3d) {
                                r = 20/255.f; g = 70/255.f; b = 235/255.f;
                            }
                        }
                    } else if (active_map_index == 2) { // Lithology
                        if (elev <= sea_level) { r = 0.08f; g = 0.08f; b = 0.31f; }
                        else {
                            switch (cell.bedrock) {
                                case RockType::BASALT: r = 0.12f; g = 0.12f; b = 0.12f; break;
                                case RockType::GRANITE: r = 0.47f; g = 0.47f; b = 0.47f; break;
                                case RockType::SANDSTONE: r = 0.82f; g = 0.70f; b = 0.55f; break;
                                case RockType::SHALE_LIMESTONE: r = 0.62f; g = 0.62f; b = 0.59f; break;
                                case RockType::METAMORPHIC: r = 0.78f; g = 0.78f; b = 1.0f; break;
                            }
                        }
                    } else if (active_map_index == 3) { // Pedology
                        if (elev <= sea_level) { r = 0.08f; g = 0.08f; b = 0.31f; }
                        else {
                            switch (cell.soil) {
                                case SoilType::NONE: r = 0.39f; g = 0.39f; b = 0.39f; break;
                                case SoilType::SAND: r = 0.94f; g = 0.90f; b = 0.55f; break;
                                case SoilType::CLAY: r = 0.70f; g = 0.39f; b = 0.19f; break;
                                case SoilType::LOAM: r = 0.31f; g = 0.15f; b = 0.08f; break;
                            }
                        }
                    } else if (active_map_index == 4) { // Hydrology
                        if (elev <= sea_level) { r = 0.06f; g = 0.06f; b = 0.23f; }
                        else if (cell.is_lake) { r = 0; g = 0.78f; b = 0.86f; }
                        else if (cell.river_flow > 0.001f) { r = 0.08f; g = 0.20f; b = 0.78f; }
                        else { r = 0.2f; g = 0.2f; b = 0.2f; }
                    } else if (active_map_index == 0) { // Biome (0)
                        if (elev <= sea_level) {
                            float depth = 1.0f - std::max(0.0f, std::min(1.0f, (elev + 10000.0f) / (sea_level + 10000.0f)));
                            if (cell.temperature < 0.2f) { r = 220/255.f; g = 240/255.f; b = 255/255.f; }
                            else {
                                r = (10 + 30 * (1.0f - depth)) / 255.0f;
                                g = (20 + 40 * (1.0f - depth)) / 255.0f;
                                b = (100 + 100 * (1.0f - depth)) / 255.0f;
                            }
                        } else if (cell.is_lake) { r = 100/255.f; g = 149/255.f; b = 237/255.f; }
                        else if (river_threshold_3d > 0.0f && cell.river_flow >= river_threshold_3d) {
                            r = 30/255.f; g = 80/255.f; b = 200/255.f; // Ríos en el globo 3D
                        }
                        else if (cell.temperature < 0.2f && !is_steep) { r = 240/255.f; g = 240/255.f; b = 240/255.f; }
                        else if (cell.temperature < 0.2f && is_steep) { r = 80/255.f; g = 80/255.f; b = 80/255.f; }
                        else {
                            biomeRGB(cell.biome, r, g, b);
                        }
                    } else if (active_map_index == 5) { // Temperature
                        float t = std::max(0.0f, std::min(1.0f, cell.temperature));
                        r = t; g = 0; b = 1.0f - t;
                    } else if (active_map_index == 6) { // Moisture
                        float m = std::max(0.0f, std::min(1.0f, cell.precipitation));
                        r = 0; g = m * 0.35f; b = m;
                    }

                    // 3. Sombreado topográfico (Hillshading) en tierra firme para dar volumen 3D
                    //    sin necesitar exageración vertical extrema.
                    if (elev > sea_level && (active_map_index == 0 || active_map_index == 1 || active_map_index == 2 || active_map_index == 3)) {
                        Vector3 grad(0.0, 0.0, 0.0);
                        for (size_t nid : cell.neighbors) {
                            Vector3 d(verts[nid].x - verts[i].x,
                                      verts[nid].y - verts[i].y,
                                      verts[nid].z - verts[i].z);
                            double dist = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + 1e-9;
                            double dh = (smooth_elev[nid] - smooth_elev[i]) / (dist * 900000.0);
                            grad.x += (d.x / dist) * dh;
                            grad.y += (d.y / dist) * dh;
                            grad.z += (d.z / dist) * dh;
                        }
                        Vector3 normal = Math::normalize(Vector3(
                            verts[i].x - grad.x,
                            verts[i].y - grad.y,
                            verts[i].z - grad.z
                        ));
                        float lambert = static_cast<float>(Math::dotProduct(normal, light_dir));
                        float shade = std::max(0.68f, std::min(1.18f, 0.88f + 0.32f * lambert));
                        r = std::min(1.0f, r * shade);
                        g = std::min(1.0f, g * shade);
                        b = std::min(1.0f, b * shade);
                    }

                    globe_vertices[i * 6 + 3] = r;
                    globe_vertices[i * 6 + 4] = g;
                    globe_vertices[i * 6 + 5] = b;
                }
                update_3d_colors = false;
            }

            // Draw 3D Globe
            if (show_3d && !globe_vertices.empty()) {
                glEnable(GL_DEPTH_TEST);
                glEnableClientState(GL_VERTEX_ARRAY);
                glEnableClientState(GL_COLOR_ARRAY);

                glVertexPointer(3, GL_FLOAT, 6 * sizeof(float), globe_vertices.data());
                glColorPointer(3, GL_FLOAT, 6 * sizeof(float), globe_vertices.data() + 3);

                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                float aspect = (float)display_w / (float)display_h;
                float f = 1.0f / tan(kCameraFovY / 2.0f);
                float zNear = 0.05f, zFar = 100.0f;
                float proj[16] = {
                    f / aspect, 0, 0, 0,
                    0, f, 0, 0,
                    0, 0, (zFar + zNear) / (zNear - zFar), -1.0f,
                    0, 0, (2.0f * zFar * zNear) / (zNear - zFar), 0
                };
                glLoadMatrixf(proj);

                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();

                glTranslatef(0.0f, 0.0f, -camera_distance);
                Vector3 col_x, col_y, col_z;
                cameraRotation(col_x, col_y, col_z);
                float rot[16] = {
                    (float)col_x.x, (float)col_x.y, (float)col_x.z, 0,
                    (float)col_y.x, (float)col_y.y, (float)col_y.z, 0,
                    (float)col_z.x, (float)col_z.y, (float)col_z.z, 0,
                    0, 0, 0, 1
                };
                glMultMatrixf(rot);

                glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(globe_indices.size()), GL_UNSIGNED_INT, globe_indices.data());

                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableClientState(GL_COLOR_ARRAY);
                glDisable(GL_DEPTH_TEST);
            }
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (sim_thread.joinable())
        sim_thread.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}