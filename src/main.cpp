#include "../include/GoldbergPolyhedron.h"
#include "../include/MapExporter.h"
#include "../include/MathUtils.h"
#include "../include/TectonicSimulator.h"
#include "../include/AtmosphereSimulator.h"
#include "../include/ErosionSimulator.h"
#include "../include/PedologySimulator.h"
#include "../include/SimulationParameters.h"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// ImGui & GLFW
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

using namespace Ravis;

// Background Simulation State
std::atomic<bool> is_simulating(false);
std::atomic<bool> new_map_ready(false);
std::mutex pixel_mutex;

std::vector<unsigned char> biome_pixels;
std::vector<unsigned char> height_pixels;
std::vector<unsigned char> lithology_pixels;
std::vector<unsigned char> pedology_pixels;

int tex_width = 1024;
int tex_height = 512;
int active_map_index = 0; // 0:Biome, 1:Height, 2:Lithology, 3:Pedology

void runSimulation(SimulationParameters params) {
    std::cout << "Starting background simulation..." << std::endl;
    GoldbergPolyhedron planet;
    
    // Level 5 (10k cells)
    planet.generate(5);
    
    TectonicSimulator tectonics(planet);
    tectonics.generatePlates(15, params);
    tectonics.simulate(100, params);

    AtmosphereSimulator atmosphere(planet);
    atmosphere.calculatePrimaryClimate(params);

    ErosionSimulator erosion(planet);
    erosion.simulateErosion(200000, params);

    atmosphere.calculateFullClimate(20, params);

    PedologySimulator pedology(planet);
    pedology.generateSoils(params);

    // Get pixels for UI
    auto b_pixels = MapExporter::getBiomePixels(planet, tex_width, tex_height, params.sea_level);
    auto h_pixels = MapExporter::getHeightPixels(planet, tex_width, tex_height, params.sea_level);
    auto l_pixels = MapExporter::getLithologyPixels(planet, tex_width, tex_height, params.sea_level);
    auto p_pixels = MapExporter::getPedologyPixels(planet, tex_width, tex_height, params.sea_level);
    
    {
        std::lock_guard<std::mutex> lock(pixel_mutex);
        biome_pixels = std::move(b_pixels);
        height_pixels = std::move(h_pixels);
        lithology_pixels = std::move(l_pixels);
        pedology_pixels = std::move(p_pixels);
        new_map_ready = true;
    }

    std::cout << "Simulation complete!" << std::endl;
    is_simulating = false;
}

int main() {
    if (!glfwInit()) return -1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Ravis World Engine - Phase 5", NULL, NULL);
    if (!window) return -1;
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
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
        
        ImGui::Text("Adjust Physical Parameters");
        ImGui::Separator();

        ImGui::SliderFloat("Sea Level", &params.sea_level, 0.1f, 0.9f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("0.0 = Dry planet. 1.0 = Waterworld. Affects evaporation and biome distribution.");

        ImGui::SliderFloat("Tectonic Speed", &params.tectonic_speed, 0.01f, 0.2f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast plates move. Faster plates create sharper collisions.");

        ImGui::SliderFloat("Orogenesis Factor", &params.orogenesis_factor, 0.1f, 3.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Multiplier for mountain height. High values create massive Himalayan ranges.");

        ImGui::SliderFloat("Erosion Rate", &params.erosion_rate, 0.01f, 1.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How fast water droplets carve through rock.");

        ImGui::SliderFloat("Global Temp Offset", &params.temp_offset, -0.5f, 0.5f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Negative values = Ice Age. Positive = Global Warming.");

        ImGui::Separator();

        if (is_simulating) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Simulating in background... Please wait.");
        } else {
            if (ImGui::Button("Generate World", ImVec2(200, 40))) {
                is_simulating = true;
                std::thread(runSimulation, params).detach();
            }
        }

        ImGui::End();

        ImGui::Begin("World Map Output");

        const char* items[] = { "Biome Map", "Height Map", "Lithology (Rock)", "Pedology (Soil)" };
        bool map_changed = ImGui::Combo("Map Layer", &active_map_index, items, IM_ARRAYSIZE(items));
        
        if (map_changed && !is_simulating && biome_pixels.size() > 0) {
            new_map_ready = true; // Trigger texture reload
        }

        if (map_texture != 0) {
            ImGui::Image((void*)(intptr_t)map_texture, ImVec2(static_cast<float>(tex_width), static_cast<float>(tex_height)));
        }
        ImGui::End();

        // Update texture if new data arrived from background thread or user changed the combo box
        if (new_map_ready) {
            std::lock_guard<std::mutex> lock(pixel_mutex);
            glBindTexture(GL_TEXTURE_2D, map_texture);
            
            if (active_map_index == 0)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, biome_pixels.data());
            else if (active_map_index == 1)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, height_pixels.data());
            else if (active_map_index == 2)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, lithology_pixels.data());
            else if (active_map_index == 3)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, pedology_pixels.data());
                
            new_map_ready = false;
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
