# Tech Context — Ravis World Engine

## Language & Standard
- **C++20** (`CMAKE_CXX_STANDARD 20`)
- **CUDA 12.x** (`CMAKE_CUDA_ARCHITECTURES 89` for RTX 1000 ADA)
- Compiler: **MinGW-w64 (GCC)** and **nvcc** on Windows

## Dependencies

| Dependency | Version | Source | Purpose |
|---|---|---|---|
| GLFW | latest | FetchContent | Window creation, input, OpenGL context |
| Dear ImGui | v1.90.1 | FetchContent | UI control panel |
| OpenGL | system | find_package | GPU rendering |
| OpenMP | system | find_package | Erosion parallelism |
| CUDA Toolkit | system | enable_language | Atmospheric SWE simulation |
| stb_image_write | vendored | `third_party/` | PNG export |

## File Structure

```
include/
├── PlanetData.h              # Cell struct, enums (BiomeType, RockType, SoilType)
├── SimulationParameters.h    # All configurable constants
├── GoldbergPolyhedron.h      # Icosahedron mesh class
├── TectonicSimulator.h       # Plate tectonics, isostasy, hotspots
├── AtmosphereSimulator.h     # Temperature, wind, moisture, biomes
├── ErosionSimulator.h        # Hydraulic erosion with OpenMP
├── HydrologySimulator.h      # Rivers, lakes, riparian effects
├── PedologySimulator.h       # Soil generation
├── MapExporter.h             # 2D equirectangular map rendering
└── MathUtils.h               # Vector math, spherical coordinates

src/
├── main.cpp                  # Entry point, ImGui loop, simulation thread
├── GoldbergPolyhedron.cpp    # Icosahedron subdivision, neighbor construction
├── TectonicSimulator.cpp     # Plate physics (subduction, convergence, divergence)
├── AtmosphereSimulator.cu    # CUDA SWE solver and Holdridge biome assignment
├── ErosionSimulator.cpp      # Droplet erosion with differential rock hardness
├── HydrologySimulator.cpp    # Flow accumulation, lake detection, riparian
├── PedologySimulator.cpp     # Soil assignment rules
└── MapExporter.cpp           # Pixel rendering with precomputed cell lookup
```

## Key Metrics
- **~10K cells** at subdivision level 5 (default)
- **~40K cells** at level 6, **~163K** at level 7
- **200K erosion droplets** (default, configurable)
- **1024×512** map resolution (default)

## Build Commands
```bash
cd build
cmake ..
cmake --build . --config Release
./RavisWorldEngine.exe
```

## Known Technical Debt
- MapExporter pixel→cell lookup is still O(n²) per build — needs spatial indexing (KD-tree) for level 7+
- `Vector3` lacks operator overloads (`+`, `-`, `*`, `dot()`, `cross()`)
- Global state in main.cpp (pixel buffers, flags) should be encapsulated in `AppState`
- Lake flood-fill doesn't check if basin connects to ocean (creates continental-scale "lakes")
