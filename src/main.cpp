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

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
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

// 3D rendering data
std::vector<float> globe_vertices; // [x,y,z, r,g,b]
std::vector<unsigned int> globe_indices;
float camera_distance = 3.0f;
float camera_yaw = 0.0f;
float camera_pitch = 0.0f;
bool is_dragging = false;
double last_mouse_x, last_mouse_y;
bool show_3d = true;
bool update_3d_geometry = false;
bool update_3d_colors = false;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) is_dragging = true;
        else if (action == GLFW_RELEASE) is_dragging = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (is_dragging) {
        camera_yaw += (xpos - last_mouse_x) * 0.01f;
        camera_pitch += (ypos - last_mouse_y) * 0.01f;
        if(camera_pitch > 1.5707f) camera_pitch = 1.5707f;
        if(camera_pitch < -1.5707f) camera_pitch = -1.5707f;
    }
    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    camera_distance -= yoffset * 0.2f;
    if (camera_distance < 1.1f) camera_distance = 1.1f;
    if (camera_distance > 10.0f) camera_distance = 10.0f;
}

void runSimulation(SimulationParameters params) {
  // If seed is 0, generate a random seed
  if (params.seed == 0) {
    params.seed = static_cast<int>(
        std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
  }

  float eff_sea = params.effective_sea_level();

  std::cout << "Starting simulation (seed: " << params.seed << ")..."
            << std::endl;
  auto new_planet = std::make_shared<GoldbergPolyhedron>();

  new_planet->generate(params.subdivision_level);

  TectonicSimulator tectonics(*new_planet);
  tectonics.generatePlates(params.num_plates, params);
  tectonics.simulate(params.tectonic_iterations(), params);

  AtmosphereSimulator atmosphere(*new_planet);
  atmosphere.calculatePrimaryClimate(params);

  ErosionSimulator erosion(*new_planet);
  int scaled_drops = static_cast<int>(params.num_drops * (params.planet_age_Myr / 500.0f));
  erosion.simulateErosion(scaled_drops, params);

  OceanSimulator ocean(*new_planet);
  ocean.simulateCurrents(params);

  atmosphere.calculateFullClimate(params.moisture_iterations, params);

  HydrologySimulator hydrology(*new_planet);
  hydrology.simulate(params);

  // Re-assign biomes after hydrology (riparian moisture changes biomes)
  atmosphere.assignBiomes(params);

  PedologySimulator pedology(*new_planet);
  pedology.generateSoils(params);

  // Build pixel->cell lookup ONCE, then render all maps in O(pixels) each
  auto cellMap =
      MapExporter::buildPixelToCellMap(*new_planet, tex_width, tex_height);
  auto b_pixels = MapExporter::getBiomePixels(*new_planet, tex_width, tex_height,
                                              eff_sea, cellMap);
  auto h_pixels = MapExporter::getHeightPixels(*new_planet, tex_width, tex_height,
                                               eff_sea, cellMap);
  auto l_pixels = MapExporter::getLithologyPixels(*new_planet, tex_width, tex_height,
                                                  eff_sea, cellMap);
  auto p_pixels = MapExporter::getPedologyPixels(*new_planet, tex_width, tex_height,
                                                 eff_sea, cellMap);
  auto y_pixels = MapExporter::getHydrologyPixels(*new_planet, tex_width, tex_height,
                                                 eff_sea, cellMap);
  auto t_pixels = MapExporter::getTemperaturePixels(*new_planet, tex_width, tex_height,
                                                 eff_sea, cellMap);
  auto m_pixels = MapExporter::getMoisturePixels(*new_planet, tex_width, tex_height,
                                                 eff_sea, cellMap);

  {
    std::lock_guard<std::mutex> lock(pixel_mutex);
    biome_pixels = std::move(b_pixels);
    height_pixels = std::move(h_pixels);
    lithology_pixels = std::move(l_pixels);
    pedology_pixels = std::move(p_pixels);
    hydrology_pixels = std::move(y_pixels);
    temperature_pixels = std::move(t_pixels);
    moisture_pixels = std::move(m_pixels);
    active_planet = new_planet;
    new_map_ready = true;
    update_3d_geometry = true;
    update_3d_colors = true;
  }

  std::cout << "Simulation complete! (seed: " << params.seed << ")"
            << std::endl;
  is_simulating = false;
}

int main() {
  if (!glfwInit())
    return -1;

  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Ravis World Engine", NULL, NULL);
  if (!window)
    return -1;

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

  SimulationParameters params;
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
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Left Click + Drag to Rotate");
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Scroll to Zoom");
        if (ImGui::Button("Reset Camera (North Up)")) {
            camera_pitch = -1.570796f;
            camera_yaw = 0.0f;
        }
    }

    if (!show_3d && map_texture != 0) {
      ImGui::Image((void *)(intptr_t)map_texture,
                   ImVec2(static_cast<float>(tex_width),
                          static_cast<float>(tex_height)));
    }
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
                        struct BP { float t, p; float r, g, b; };
                        BP biomes[] = {
                            {0.2f, 0.1f, 160/255.f, 160/255.f, 120/255.f}, {0.2f, 0.4f, 140/255.f, 150/255.f, 100/255.f}, {0.3f, 0.7f, 90/255.f, 120/255.f, 90/255.f}, {0.3f, 0.9f, 50/255.f, 90/255.f, 60/255.f},
                            {0.5f, 0.1f, 210/255.f, 180/255.f, 140/255.f}, {0.6f, 0.3f, 180/255.f, 180/255.f, 90/255.f}, {0.5f, 0.6f, 120/255.f, 180/255.f, 90/255.f}, {0.6f, 0.9f, 34/255.f, 139/255.f, 34/255.f},
                            {0.9f, 0.1f, 237/255.f, 201/255.f, 175/255.f}, {0.8f, 0.3f, 200/255.f, 180/255.f, 100/255.f}, {0.9f, 0.5f, 154/255.f, 205/255.f, 50/255.f}, {0.8f, 0.7f, 100/255.f, 160/255.f, 40/255.f}, {0.9f, 0.9f, 0/255.f, 100/255.f, 0/255.f}
                        };
                        float min_dist = 1e10f;
                        for (const auto& bp : biomes) {
                            float dt = cell.temperature - bp.t;
                            float dp = cell.precipitation - bp.p;
                            float dist = dt*dt + dp*dp;
                            if (dist < min_dist) {
                                min_dist = dist;
                                r = bp.r; g = bp.g; b = bp.b;
                            }
                        }
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
            float fov = 45.0f * 3.1415926535f / 180.0f;
            float f = 1.0f / tan(fov / 2.0f);
            float zNear = 0.1f, zFar = 100.0f;
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
            // Rotate
            glRotatef(camera_pitch * 180.0f / 3.1415926535f, 1.0f, 0.0f, 0.0f);
            glRotatef(camera_yaw * 180.0f / 3.1415926535f, 0.0f, 1.0f, 0.0f);
            
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
