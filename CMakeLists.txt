cmake_minimum_required(VERSION 3.20)
project(RavisWorldEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Add include directories
include_directories(include)
include_directories(third_party)

# Add source files
set(SOURCES
    src/main.cpp
    src/GoldbergPolyhedron.cpp
    src/MapExporter.cpp
    src/MapExporterCUDA.cu
    src/TectonicSimulator.cu
    src/AtmosphereSimulator.cu
    src/OceanSimulator.cu
    src/ErosionSimulator.cu
    src/PedologySimulator.cpp
    src/HydrologySimulator.cpp
)

# Find OpenMP for erosion parallelism
find_package(OpenMP REQUIRED)

# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

# Add ImGui source files explicitly since it's typically distributed as source
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Create executable
add_executable(RavisWorldEngine ${SOURCES} ${IMGUI_SOURCES})

# Include directories for ImGui
target_include_directories(RavisWorldEngine PRIVATE 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
)

# Find OpenGL (cross-platform)
find_package(OpenGL REQUIRED)

# Link OpenMP, GLFW and OpenGL
target_link_libraries(RavisWorldEngine PRIVATE 
    OpenMP::OpenMP_CXX 
    glfw 
    OpenGL::GL
)
