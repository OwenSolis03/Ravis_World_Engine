# Progress — Ravis World Engine

## Completed Phases

### Phase 1–3: Core Engine ✅
- Goldberg Polyhedron (subdivided icosahedron)
- Tectonic plates (BFS Voronoi, convergence/divergence)
- Basic heightmap, temperature, precipitation
- PNG map export

### Phase 4: Erosion ✅
- Hydraulic erosion (200K droplets, OpenMP parallelized)
- Differential erosion by rock type
- Sedimentary rock deposition

### Phase 5: Tooling & UI ✅
- Dear ImGui control panel
- 13 configurable parameters with tooltips
- Background simulation thread
- Map layer selector (5 modes)

### Phase 6: Geology ✅
- 5 rock types (Basalt, Granite, Sandstone, Shale/Limestone, Metamorphic)
- 4 soil types (Sand, Clay, Loam, None)
- Metamorphism at extreme elevation

## Completed Improvements (This Session)

### Elevation & Geography Refactor ✅
- `height` (0-1) → `elevation` (meters). Sea level at 0m.
- Isostasy formula corrected to provide thick crustal roots for high mountains, preventing sinking.
- **Fractal Ridge Blending**: Applied absolute value 3D noise to create sharp mountain ridges instead of rolling plains.

### Tectonics & Volcanism ✅
- **Continental Clustering**: Added Primary (Pangea) and Secondary (Americas) clustering physics.
- **Crust Warping**: Added non-separable hash noise distortion for erratic landmass shapes.
- Tectonic plates differentiated: Oceanic plates move 3x faster than Continental plates.
- Collision mechanics refined: Cont-Cont produces extreme orogenesis (Himalayas), Oceanic-Oceanic creates island arcs.
- **Organic Islands**: Differentiated plumes into Deep (Iceland) and Shallow (Hawaii), using 3D FBM hash-noise to distort circular shapes into jagged islands.
- **Superswells**: Massive African-style mantle uplifts affecting both oceanic and continental crusts.
- **Inverse Mapping (Backward Advection)**: Rewrote tectonic advection using inverse Rodrigues Rotation backward in time to perfectly resolve discrete coordinate mapping errors and prevent continent disintegration.
- **Velocity Tuning**: Scaled down maximum plate rotation velocities to ~0.015 rad/step to maintain coherent landmass shapes over long epochs.

### Geophysics & Terrain Features ✅
- Replaced separable trigonometric noise with **Hash-Based 3D Noise** (via irrational vector dot products) to permanently eradicate mirror symmetry. Added FBM support.
- Added Appalachian-style Old Mountains, Caledonian-style Old Hills, Volga-style Uplands, and stochastic small uplifts using radial BFS with noise modulation.

### Climate & Hydrology ✅
- **Paleoclimate**: Integrated LGM (Last Glacial Maximum) temperature anomalies and post-LGM sea level rises.
- **CUDA Atmosphere**: Shallow Water Equations for winds and Holdridge Life Zones for ideal biomes.
- **Voronoi Biomes**: 3% land seeds used to flood-fill biomes, followed by Cellular Automata smoothing to break up mega-deserts and create organic biome clusters.
- **BFS Rivers**: HydrologySimulator routes rivers out of endorheic basins (pits) using Breadth-First Search, carving massive canyons to the sea and eradicating continent-spanning lakes.

### Map Rendering ✅
- MapExporter exact UI color match implemented for elevation rendering.
- Hydrology map dry land set to precise dark gray (0.2, 0.2, 0.2) to emphasize rivers and lakes.

### Phase 2: Tectonics Overhaul & Continuous Geometry ✅
- **Oceanic vs Continental Crust**: Explicit `is_oceanic` tagging. Oceanic plates sink to form deep basins and trenches upon subduction.
- **Orogenesis Dynamics**: Frontal collisions form massive continental ridges (Himalayas) using clamped FBM and `hash_ridge3d`, while subduction creates trenches.
- **Continuous Global FBM**: Removed per-plate noise offsets to ensure a completely seamless and mathematically continuous height gradient across the globe, eradicating giant boundary cliffs and tearing.
- **Bisector Boundary Math**: Added UI toggle for exact mathematical distance-to-fault stress calculation vs cellular diffusion.
- **Precipitation Baseline**: Fixed moisture physics to enforce a baseline evaporation/rain cycle on flat plains, permanently solving the "megadesert domination" issue.
- **Steep Slope Masking (3D)**: Implemented local gradient slope calculation in the 3D renderer to override snow with dark rock on vertical cliffs, adding realism to mountains and poles.

## Pending / Known Issues
- [ ] Vector3 lacks operator overloads
- [ ] main.cpp globals should be encapsulated in AppState
- [ ] Pixel→cell lookup needs KD-tree for level 7+
