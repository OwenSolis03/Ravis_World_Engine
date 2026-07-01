# Product Context — Ravis World Engine

## What It Does
Generates procedurally-simulated planets by running a multi-stage physical simulation pipeline, then renders 2D equirectangular maps of the results.

## How It Works (Pipeline)
1. **Goldberg Polyhedron** → Subdivided icosahedron mesh (~10K hexagonal cells at level 5)
2. **Tectonics** → BFS plate assignment, convergent/divergent boundary physics, isostasy, hotspot volcanism
3. **Primary Climate** → Latitude-based temperature, initial moisture
4. **Hydraulic Erosion** → 200K droplets carving terrain via differential rock hardness (OpenMP parallelized)
5. **Full Climate** → Wind circulation (Coriolis), multi-neighbor moisture advection, orographic precipitation, biome assignment
6. **Hydrology** → Flow accumulation rivers, endorheic basin lakes, riparian moisture boost, biome re-assignment
7. **Pedology** → Soil type assignment based on bedrock, climate, and biome
8. **Map Export** → Precomputed pixel→cell lookup, 5 map layers (Biome, Height, Lithology, Pedology, Hydrology)

## User Interaction
- Dear ImGui floating control panel with categorized sliders
- "Generate World" button launches simulation in background thread
- Map Layer combo box to switch between 5 visualization modes
- Seed input (0 = random) + "Randomize" button for variety

## Key Design Decisions
- **Elevation in meters** (not normalized 0-1) for physical accuracy
- **Sea level at 0m** by default, configurable ±500m
- **Configurable seed** for reproducibility
- All simulation constants exposed as `SimulationParameters` struct
