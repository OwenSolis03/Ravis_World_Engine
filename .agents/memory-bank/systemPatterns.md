# System Patterns — Ravis World Engine

## Architecture
- **Monolithic executable** with CMake build system (C++20, CUDA, MinGW on Windows)
- **Header/Source separation**: `include/*.h` + `src/*.cpp`/`*.cu`
- **Each simulator is a class** that takes a `GoldbergPolyhedron&` reference and mutates its cells in-place
- **Pipeline execution**: Simulators run sequentially in `runSimulation()` — order matters for physical dependencies

## Data Model
- `Cell` struct in `PlanetData.h` — the universal data carrier with ~20 fields covering geology, climate, hydrology
- `SimulationParameters` — flat struct with all configurable constants, passed by value to the simulation thread
- `GoldbergPolyhedron` — owns the `std::vector<Cell>`, provides `getCells()` accessor

## Concurrency Model
- **Main thread**: ImGui render loop (GLFW + OpenGL)
- **Simulation thread**: Single `std::thread` stored as `sim_thread`, joined on cleanup
- **Erosion**: OpenMP `#pragma omp parallel for` with `#pragma omp atomic` for elevation and `#pragma omp critical` for bedrock
- **Atmosphere**: CUDA kernels (`wind_update_kernel`, `pressure_update_kernel`) simulating SWE on the GPU
- **Thread safety**: `std::mutex pixel_mutex` guards pixel buffer transfer; `std::atomic<bool>` for flags

## Key Patterns
- **Pixel→Cell Lookup**: `MapExporter::buildPixelToCellMap()` precomputes the O(n²) nearest-cell search once, shared across all 5 renderers
- **Steepest Descent Drainage**: Cells sorted by elevation (high→low), flow accumulated downstream via `downstream_id`
- **Selective Smoothing**: Only plate interiors are smoothed (every 5 steps), boundaries preserved for sharp features
- **Riparian Effect**: Rivers/lakes boost adjacent cell moisture → biomes re-assigned after hydrology

## Build System
- CMake 3.20+, C++20, CUDA Toolkit
- FetchContent: GLFW, ImGui (v1.90.1)
- find_package: OpenMP, OpenGL
- Third-party (vendored): stb_image_write.h
