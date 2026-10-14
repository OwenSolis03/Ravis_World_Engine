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

std::vector<unsigned char> biome_pixels;
std::vector<unsigned char> height_pixels;
std::vector<unsigned char> lithology_pixels;
std::vector<unsigned char> pedology_pixels;
std::vector<unsigned char> hydrology_pixels;
std::vector<unsigned char> temperature_pixels;
std::vector<unsigned char> moisture_pixels;

int tex_width = 1024;
int tex_height = 512;
int active_map_index = 0; // 0:Biome, 1:Height, 2:Lithology, 3:Pedology, 4:Hydrology

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
bool show_3d = true;
bool update_3d_geometry = false;
bool update_3d_colors = false;
const float kCameraFovY = 45.0f * 3.1415926535f / 180.0f;
const float kCameraMinDistance = 1.15f;
const float kCameraMaxDistance = 8.0f;

// 2D map editing state
std::vector<size_t> map_cell_lookup; // pixel -> cell of the current planet
bool is_painting_2d = false;
bool map_preview_dirty = false; // elevation changed under the 2D map

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

// Equirectangular map pixel <-> unit-sphere point. Same convention as the
// pixel->cell lookup: u in [0,1] spans lon -pi..pi, v in [0,1] spans lat
// +pi/2..-pi/2.
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

// Rebuild all 2D map textures from the planet and publish it as the active
// world (shared by full generation and by post-edit recomputes).
static void publishMaps(std::shared_ptr<GoldbergPolyhedron> planet,
                        const SimulationParameters& p) {
    float eff_sea = p.effective_sea_level();
    auto cellMap = MapExporter::buildPixelToCellMap(*planet, tex_width, tex_height);
    auto b_pixels = MapExporter::getBiomePixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto h_pixels = MapExporter::getHeightPixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto l_pixels = MapExporter::getLithologyPixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto p_pixels = MapExporter::getPedologyPixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto y_pixels = MapExporter::getHydrologyPixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto t_pixels = MapExporter::getTemperaturePixels(*planet, tex_width, tex_height, eff_sea, cellMap);
    auto m_pixels = MapExporter::getMoisturePixels(*planet, tex_width, tex_height, eff_sea, cellMap);

    std::lock_guard<std::mutex> lock(pixel_mutex);
    biome_pixels = std::move(b_pixels);
    height_pixels = std::move(h_pixels);
    lithology_pixels = std::move(l_pixels);
    pedology_pixels = std::move(p_pixels);
    hydrology_pixels = std::move(y_pixels);
    temperature_pixels = std::move(t_pixels);
    moisture_pixels = std::move(m_pixels);
    map_cell_lookup = std::move(cellMap);
    active_planet = planet;
    new_map_ready = true;
    update_3d_geometry = true;
    update_3d_colors = true;
}

// Wall-clock time per pipeline stage: lap("name") closes the stage that just
// ran; report() prints the table and appends it to worldgen.log.
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

// Re-derive climate -> hydrology -> biomes -> soils from the (edited)
// elevation field, without regenerating the mesh or re-running tectonics and
// erosion. Erosion is intentionally excluded so it cannot eat user edits;
// this is the same pipeline order runSimulation uses after erosion.
void recomputeDerivedFields(SimulationParameters p) {
    std::shared_ptr<GoldbergPolyhedron> planet;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        planet = active_planet;
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

// Apply one brush dab centered on the unit-sphere point p. `cell` is the cell
// under p when the caller already knows it (2D map lookup), or -1 to search.
// Runs on the main thread; never while a background simulation owns the cells.
static void paintAtPoint(const Vector3& p, int cell) {
    if (is_simulating) return;
    std::shared_ptr<GoldbergPolyhedron> planet;
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        planet = active_planet;
    }
    if (!planet || !terrain_editor.hasBase()) return;

    if (cell < 0) cell = TerrainEditor::pickCell(*planet, p);
    if (cell < 0 || cell == last_painted_cell) return; // one dab per cell crossed
    last_painted_cell = cell;

    float radius_rad = brush_radius_km / kPlanetRadiusKm;
    EditOpType type = (brush_mode == 0) ? EditOpType::RAISE : EditOpType::LOWER;
    if (terrain_editor.applyDab(*planet, type, p, radius_rad, brush_strength_m)) {
        stroke_made_changes = true;
        update_3d_colors = true;  // show the raw terrain change immediately
        map_preview_dirty = true; // ...and on the 2D map
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
    // Releases are always handled: a press on the globe followed by a release
    // over a UI panel must still end the drag / stroke.
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
                // Propagate the edit: winds, rain shadow, rivers, biomes.
                if (stroke_made_changes) startRecompute();
            }
        }
    }
    // Right button always rotates, so the camera still works in Edit Mode.
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
        // Grab-and-drag: the surface under the cursor follows the mouse, so
        // the angle per pixel shrinks as the camera gets closer.
        int ww = 0, wh = 0;
        glfwGetWindowSize(window, &ww, &wh);
        float rad_per_px = 2.0f * tanf(kCameraFovY * 0.5f) *
                           (camera_distance - 1.0f) / static_cast<float>(std::max(wh, 1));
        float lon_scale = 1.0f / std::max(cosf(camera_lat), 0.2f);
        camera_lon -= static_cast<float>(xpos - last_mouse_x) * rad_per_px * lon_scale;
        camera_lat += static_cast<float>(ypos - last_mouse_y) * rad_per_px;

        const float pi = 3.1415926535f;
        const float max_lat = pi * 0.5f - 0.01f; // stop just short of the poles
        camera_lat = std::max(-max_lat, std::min(max_lat, camera_lat));
        if (camera_lon > pi) camera_lon -= 2.0f * pi;
        if (camera_lon < -pi) camera_lon += 2.0f * pi;
    }
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    // Zoom scales the height above the surface, so each step feels the same
    // far away and close up.
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

// Dump the parameters a world was generated with + summary diagnostics to
// stdout and append them to worldgen.log (in the working directory). The
// diagnostics exist to tune terrain/climate — e.g. mean continental elevation
// is the number to watch when continents come out too tall.
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
  o << "    approx mm/yr   mean   " << (land ? (pSumLand / land) * 8000.0 : 0.0)
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
  // If seed is 0, generate a random seed
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
  pedology.classifyBedrock(params); // refine rock types from relief + climate
  pedology.generateSoils(params);
  timer.lap("biomes + soils");

  // Record parameters + diagnostics for this world (stdout + worldgen.log)
  logWorldStats(*new_planet, params);

  // The finished terrain becomes the immutable base layer for brush edits.
  // Locked because the UI thread reads the editor's stroke counters.
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
  // --headless [subdivision]: generate one world with the default parameters,
  // print the stats and stage timings, and exit without opening a window.
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

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Ravis World Engine", NULL, NULL);
  if (!window)
    return -1;

  g_window = window;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // VSync

  // Set callbacks
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

  GLuint map_texture = 0;

  // Create an empty texture initially
  glGenTextures(1, &map_texture);
  glBindTexture(GL_TEXTURE_2D, map_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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
      ImGui::SetTooltip(
          "0 = Random seed each generation. Any other value = deterministic.");
    ImGui::SameLine();
    if (ImGui::SmallButton("Randomize")) {
      params.seed = static_cast<int>(
          std::chrono::steady_clock::now().time_since_epoch().count() %
          1000000);
    }

    ImGui::SliderInt("Subdivision", &params.subdivision_level, 3, 8);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Mesh resolution. (5 = 10K, 6 = 40K, 7 = 163K, 8 = 655K cells) "
                        "Higher = slower.");
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
      show_3d = false; // editing happens on the 2D map by default
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

    ImGui::Begin("World Map Output");

    const char *items[] = {"Biome Map", "Height Map", "Lithology (Rock)",
                           "Pedology (Soil)", "Hydrology", "Temperature", "Moisture"};
    bool map_changed = ImGui::Combo("Map Layer", &active_map_index, items,
                                    IM_ARRAYSIZE(items));

    if (map_changed && !is_simulating) {
      new_map_ready = true;
      update_3d_colors = true;
    }

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
      if (edit_mode)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Left Click + Drag on the map to Paint");

      // Fit the 2:1 map to the window width.
      float map_w = std::max(256.0f, ImGui::GetContentRegionAvail().x);
      ImVec2 map_size(map_w, map_w * 0.5f);
      ImVec2 map_min = ImGui::GetCursorScreenPos();
      ImGui::Image((void *)(intptr_t)map_texture, map_size);
      // An invisible button over the image takes the mouse, so dragging on the
      // map paints instead of moving the window.
      ImGui::SetCursorScreenPos(map_min);
      ImGui::InvisibleButton("map_canvas", map_size);
      bool map_hovered = ImGui::IsItemHovered();

      std::shared_ptr<GoldbergPolyhedron> map_planet;
      {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        if (!is_simulating) map_planet = active_planet;
      }
      bool lookup_ok = map_planet && map_cell_lookup.size() ==
                       static_cast<size_t>(tex_width) * tex_height;

      if (map_hovered && lookup_ok) {
        ImVec2 mouse = ImGui::GetMousePos();
        double u = std::max(0.0, std::min(1.0, (double)(mouse.x - map_min.x) / map_size.x));
        double v = std::max(0.0, std::min(1.0, (double)(mouse.y - map_min.y) / map_size.y));
        int px = static_cast<int>(u * (tex_width - 1) + 0.5);
        int py = static_cast<int>(v * (tex_height - 1) + 0.5);
        size_t cell_id = map_cell_lookup[static_cast<size_t>(py) * tex_width + px];
        Vector3 p = mapUVToSphere(u, v);

        const Cell& hovered = map_planet->getCells()[cell_id];
        ImGui::Text("Lat %.1f  Lon %.1f  Elevation %.0f m",
                    hovered.latitude * 180.0 / Math::PI,
                    hovered.longitude * 180.0 / Math::PI,
                    hovered.elevation - params.effective_sea_level());

        if (edit_mode && terrain_editor.hasBase()) {
          // Brush outline: the geodesic circle of the brush, which is not a
          // circle on an equirectangular map (it widens towards the poles).
          double radius_rad = brush_radius_km / kPlanetRadiusKm;
          Vector3 ref = std::abs(p.z) < 0.9 ? Vector3(0, 0, 1) : Vector3(1, 0, 0);
          Vector3 a = Math::normalize(Math::crossProduct(p, ref));
          Vector3 b = Math::crossProduct(p, a);
          ImDrawList* draw = ImGui::GetWindowDrawList();
          draw->PushClipRect(map_min, ImVec2(map_min.x + map_size.x, map_min.y + map_size.y), true);
          const int kSegments = 64;
          ImVec2 prev;
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
            ImVec2 cur(map_min.x + (float)qu * map_size.x, map_min.y + (float)qv * map_size.y);
            // Skip the segment that jumps across the +-180 seam.
            if (s > 0 && std::abs(cur.x - prev.x) < map_size.x * 0.5f)
              draw->AddLine(prev, cur, IM_COL32(255, 255, 255, 220), 1.5f);
            prev = cur;
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
        // Propagate the edit: winds, rain shadow, rivers, biomes.
        if (stroke_made_changes) startRecompute();
      }

      // Show the raw terrain change on the active layer while the stroke is in
      // progress; the full recompute on release refreshes every layer.
      if (map_preview_dirty && lookup_ok) {
        float eff_sea = params.effective_sea_level();
        std::vector<unsigned char>* layers[] = {
            &biome_pixels, &height_pixels, &lithology_pixels, &pedology_pixels,
            &hydrology_pixels, &temperature_pixels, &moisture_pixels};
        std::vector<unsigned char> preview;
        switch (active_map_index) {
        case 0: preview = MapExporter::getBiomePixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        case 1: preview = MapExporter::getHeightPixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        case 2: preview = MapExporter::getLithologyPixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        case 3: preview = MapExporter::getPedologyPixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        case 4: preview = MapExporter::getHydrologyPixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        case 5: preview = MapExporter::getTemperaturePixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        default: preview = MapExporter::getMoisturePixels(*map_planet, tex_width, tex_height, eff_sea, map_cell_lookup); break;
        }
        std::lock_guard<std::mutex> lock(pixel_mutex);
        *layers[active_map_index] = std::move(preview);
        new_map_ready = true;
      }
    }
    map_preview_dirty = false;
    ImGui::End();

    // Legends & Symbology Panel
    ImGui::Begin("Legends & Symbology");
    if (active_map_index == 0) {
        ImGui::Text("Biome Colors");
        ImGui::Separator();
        const ImVec4 colors[] = {
            ImVec4(0,0,1,1), ImVec4(0.8,0.9,1,1), ImVec4(1,1,1,1), ImVec4(0.5,0.7,0.5,1),
            ImVec4(0.2,0.6,0.2,1), ImVec4(0,0.4,0,1), ImVec4(0.7,0.7,0.3,1), ImVec4(0.4,0.7,0.2,1),
            ImVec4(0.1,0.5,0.1,1), ImVec4(0.9,0.8,0.6,1), ImVec4(0.8,0.7,0.4,1), ImVec4(0.6,0.8,0.2,1),
            ImVec4(0.2,0.8,0.2,1), ImVec4(1,0.9,0.7,1), ImVec4(0.6,0.4,0.2,1)
        };
        const char* names[] = {
            "Ocean", "Ice", "Tundra", "Boreal Forest", "Temperate Deciduous",
            "Temperate Rainforest", "Steppe", "Woodland", "Tropical Seasonal",
            "Desert", "Tropical Dry Forest", "Savanna", "Tropical Rainforest",
            "Mediterranean", "Bare Rock"
        };
        for(int i = 0; i < 15; i++) {
            ImGui::ColorButton("##", colors[i], ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker);
            ImGui::SameLine();
            ImGui::Text("%s", names[i]);
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
        ImGui::ColorButton("##1", ImVec4(0.2,0.2,0.2,1)); ImGui::SameLine(); ImGui::Text("Basalt (Oceanic Crust)");
        ImGui::ColorButton("##2", ImVec4(0.8,0.7,0.7,1)); ImGui::SameLine(); ImGui::Text("Granite (Continental Craton)");
        ImGui::ColorButton("##3", ImVec4(0.8,0.8,0.4,1)); ImGui::SameLine(); ImGui::Text("Sandstone (Sedimentary)");
        ImGui::ColorButton("##4", ImVec4(0.6,0.6,0.6,1)); ImGui::SameLine(); ImGui::Text("Shale / Limestone (Marine Deposit)");
        ImGui::ColorButton("##5", ImVec4(0.5,0.2,0.5,1)); ImGui::SameLine(); ImGui::Text("Metamorphic (Orogenesis Core)");
    } else if (active_map_index == 3) {
        ImGui::Text("Pedology (Soil Type)");
        ImGui::Separator();
        ImGui::ColorButton("##1", ImVec4(0.8,0.7,0.5,1)); ImGui::SameLine(); ImGui::Text("Aridisol (Desert Sand)");
        ImGui::ColorButton("##2", ImVec4(0.3,0.2,0.1,1)); ImGui::SameLine(); ImGui::Text("Mollisol (Fertile Grassland)");
        ImGui::ColorButton("##3", ImVec4(0.6,0.2,0.1,1)); ImGui::SameLine(); ImGui::Text("Oxisol (Tropical Rust)");
        ImGui::ColorButton("##4", ImVec4(0.4,0.4,0.4,1)); ImGui::SameLine(); ImGui::Text("Gelisol (Permafrost)");
        ImGui::ColorButton("##5", ImVec4(0.5,0.4,0.3,1)); ImGui::SameLine(); ImGui::Text("Inceptisol (Young Soil)");
    } else if (active_map_index == 4) {
        ImGui::Text("Hydrology");
        ImGui::Separator();
        ImGui::ColorButton("##1", ImVec4(0.2,0.2,0.8,1)); ImGui::SameLine(); ImGui::Text("Ocean");
        ImGui::ColorButton("##2", ImVec4(0.4,0.8,1.0,1)); ImGui::SameLine(); ImGui::Text("Lake / Riparian");
        ImGui::ColorButton("##3", ImVec4(0.0,0.0,0.5,1)); ImGui::SameLine(); ImGui::Text("River (High Flow)");
        ImGui::ColorButton("##4", ImVec4(0.2,0.2,0.2,1)); ImGui::SameLine(); ImGui::Text("Dry Land");
    } else if (active_map_index == 5) {
        ImGui::Text("Temperature");
        ImGui::Separator();
        ImGui::ColorButton("##1", ImVec4(1,0,0,1)); ImGui::SameLine(); ImGui::Text("Hot (Equatorial)");
        ImGui::ColorButton("##2", ImVec4(0.5,0,0.5,1)); ImGui::SameLine(); ImGui::Text("Moderate (Temperate)");
        ImGui::ColorButton("##3", ImVec4(0,0,1,1)); ImGui::SameLine(); ImGui::Text("Cold (Polar)");
    } else if (active_map_index == 6) {
        ImGui::Text("Moisture");
        ImGui::Separator();
        ImGui::ColorButton("##1", ImVec4(0,0,1,1)); ImGui::SameLine(); ImGui::Text("Wet (Rainforest)");
        ImGui::ColorButton("##2", ImVec4(0,0,0.5,1)); ImGui::SameLine(); ImGui::Text("Moderate");
        ImGui::ColorButton("##3", ImVec4(0,0,0,1)); ImGui::SameLine(); ImGui::Text("Dry (Desert)");
    }
    ImGui::End();

    // Update texture if new data arrived
    if (new_map_ready) {
      std::lock_guard<std::mutex> lock(pixel_mutex);
      glBindTexture(GL_TEXTURE_2D, map_texture);

      if (active_map_index == 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, biome_pixels.data());
      else if (active_map_index == 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, height_pixels.data());
      else if (active_map_index == 2)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, lithology_pixels.data());
      else if (active_map_index == 3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, pedology_pixels.data());
      else if (active_map_index == 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, hydrology_pixels.data());
      else if (active_map_index == 5)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, temperature_pixels.data());
      else if (active_map_index == 6)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, moisture_pixels.data());

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

    if (current_planet) {
        if (update_3d_geometry) {
            const auto& verts = current_planet->getVertices();
            const auto& tris = current_planet->getTriangles();
            globe_vertices.resize(verts.size() * 6);
            for (size_t i = 0; i < verts.size(); ++i) {
                globe_vertices[i * 6 + 0] = verts[i].x;
                globe_vertices[i * 6 + 1] = verts[i].y;
                globe_vertices[i * 6 + 2] = verts[i].z;
                globe_vertices[i * 6 + 3] = 1.0f;
                globe_vertices[i * 6 + 4] = 1.0f;
                globe_vertices[i * 6 + 5] = 1.0f;
            }
            globe_indices.resize(tris.size() * 3);
            for (size_t i = 0; i < tris.size(); ++i) {
                globe_indices[i * 3 + 0] = tris[i].v[0];
                globe_indices[i * 3 + 1] = tris[i].v[1];
                globe_indices[i * 3 + 2] = tris[i].v[2];
            }
            update_3d_geometry = false;
        }

        if (update_3d_colors && !globe_vertices.empty()) {
            const auto& cells = current_planet->getCells();
            const auto& verts = current_planet->getVertices();
            for (size_t i = 0; i < cells.size(); ++i) {
                float r=1.0f, g=1.0f, b=1.0f;
                const auto& cell = cells[i];
                float elev = cell.elevation;
                float sea_level = params.effective_sea_level();

                // Displace vertex based on elevation
                float scale = 1.0f;
                if (elev > sea_level) {
                    scale = 1.0f + (elev - sea_level) * 0.000005f; // Mountains stick out
                } else {
                    scale = 1.0f + (elev - sea_level) * 0.000002f; // Ocean trenches sink in
                }
                globe_vertices[i * 6 + 0] = verts[i].x * scale;
                globe_vertices[i * 6 + 1] = verts[i].y * scale;
                globe_vertices[i * 6 + 2] = verts[i].z * scale;

                // Calculate slope for steep rock masking (Snow shouldn't stick to vertical cliffs)
                float max_slope = 0.0f;
                for (size_t nid : cell.neighbors) {
                    float diff = fabsf(cells[nid].elevation - elev);
                    if (diff > max_slope) max_slope = diff;
                }
                bool is_steep = max_slope > 1000.0f; // Threshold for steep cliff

                if (active_map_index == 1) { // Height
                    if (elev <= sea_level) {
                        float depth = std::max(0.0f, std::min(1.0f, (elev + 11000.0f) / (sea_level + 11000.0f)));
                        r = (0) / 255.0f;
                        g = (100 * depth) / 255.0f;
                        b = (50 + 150 * depth) / 255.0f;
                    } else {
                        float land = (elev - sea_level);
                        if (land < 1000.0f) {
                            float t = land / 1000.0f;
                            r = (34 * (1-t) + 173 * t) / 255.0f;
                            g = (139 * (1-t) + 255 * t) / 255.0f;
                            b = (34 * (1-t) + 47 * t) / 255.0f;
                        } else if (land < 3000.0f) {
                            float t = (land - 1000.0f) / 2000.0f;
                            r = (173 * (1-t) + 139 * t) / 255.0f;
                            g = (255 * (1-t) + 69 * t) / 255.0f;
                            b = (47 * (1-t) + 19 * t) / 255.0f;
                        } else {
                            float t = std::min(1.0f, (land - 3000.0f) / 5000.0f);
                            r = (139 * (1-t) + 60 * t) / 255.0f;
                            g = (69 * (1-t) + 30 * t) / 255.0f;
                            b = (19 * (1-t) + 10 * t) / 255.0f;
                        }
                    }
                    if (cell.temperature < 0.2f && !is_steep && elev > sea_level) { // Ice overlay, but not on steep cliffs
                        r = 1.0f; g = 1.0f; b = 1.0f;
                    } else if (cell.temperature < 0.2f && elev <= sea_level) {
                        r = 1.0f; g = 1.0f; b = 1.0f; // Sea ice doesn't care about slope
                    }
                } else if (active_map_index == 2) { // Lithology
                    if (elev <= sea_level) { r=0.08f; g=0.08f; b=0.31f; }
                    else {
                        switch (cell.bedrock) {
                            case RockType::BASALT: r=0.12f; g=0.12f; b=0.12f; break;
                            case RockType::GRANITE: r=0.47f; g=0.47f; b=0.47f; break;
                            case RockType::SANDSTONE: r=0.82f; g=0.70f; b=0.55f; break;
                            case RockType::SHALE_LIMESTONE: r=0.62f; g=0.62f; b=0.59f; break;
                            case RockType::METAMORPHIC: r=0.78f; g=0.78f; b=1.0f; break;
                        }
                    }
                } else if (active_map_index == 3) { // Pedology
                    if (elev <= sea_level) { r=0.08f; g=0.08f; b=0.31f; }
                    else {
                        switch (cell.soil) {
                            case SoilType::NONE: r=0.39f; g=0.39f; b=0.39f; break;
                            case SoilType::SAND: r=0.94f; g=0.90f; b=0.55f; break;
                            case SoilType::CLAY: r=0.70f; g=0.39f; b=0.19f; break;
                            case SoilType::LOAM: r=0.31f; g=0.15f; b=0.08f; break;
                        }
                    }
                } else if (active_map_index == 4) { // Hydrology
                    if (elev <= sea_level) { r=0.06f; g=0.06f; b=0.23f; }
                    else if (cell.is_lake) { r=0; g=0.78f; b=0.86f; }
                    else if (cell.river_flow > 0.001f) { r=0.08f; g=0.20f; b=0.78f; } // Approximation without global sort
                    else { r=0.2f; g=0.2f; b=0.2f; }
                } else if (active_map_index == 0) { // Biome (0)
                    if (elev <= sea_level) {
                        float depth = 1.0f - std::max(0.0f, std::min(1.0f, (elev + 10000.0f) / (sea_level + 10000.0f)));
                        if (cell.temperature < 0.2f) { r=220/255.f; g=240/255.f; b=255/255.f; } // Sea ice
                        else {
                            r = (10 + 30 * (1.0f - depth)) / 255.0f;
                            g = (20 + 40 * (1.0f - depth)) / 255.0f;
                            b = (100 + 100 * (1.0f - depth)) / 255.0f;
                        }
                    } else if (cell.is_lake) { r=100/255.f; g=149/255.f; b=237/255.f; }
                    else if (cell.temperature < 0.2f && !is_steep) { r=240/255.f; g=240/255.f; b=240/255.f; } // Snow
                    else if (cell.temperature < 0.2f && is_steep) { r=80/255.f; g=80/255.f; b=80/255.f; } // Exposed steep rock
                    else {
                        // Whittaker biome assigned by AtmosphereSimulator::assignBiomes
                        biomeRGB(cell.biome, r, g, b);
                    }
                } else if (active_map_index == 5) { // Temperature
                    float t = std::max(0.0f, std::min(1.0f, cell.temperature));
                    r = t; g = 0; b = 1.0f - t;
                } else if (active_map_index == 6) { // Moisture
                    float m = std::max(0.0f, std::min(1.0f, cell.precipitation));
                    r = 0; g = 0; b = m;
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
            
            // Translate back
            glTranslatef(0.0f, 0.0f, -camera_distance);
            // Rotate (column-major, same matrix pickSpherePoint inverts)
            Vector3 col_x, col_y, col_z;
            cameraRotation(col_x, col_y, col_z);
            float rot[16] = {
                (float)col_x.x, (float)col_x.y, (float)col_x.z, 0,
                (float)col_y.x, (float)col_y.y, (float)col_y.z, 0,
                (float)col_z.x, (float)col_z.y, (float)col_z.z, 0,
                0, 0, 0, 1
            };
            glMultMatrixf(rot);

            // Render
            glDrawElements(GL_TRIANGLES, globe_indices.size(), GL_UNSIGNED_INT, globe_indices.data());

            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisable(GL_DEPTH_TEST);
        }
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Wait for simulation thread to finish before destroying resources
  if (sim_thread.joinable())
    sim_thread.join();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
