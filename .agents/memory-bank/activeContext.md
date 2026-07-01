# Active Context — Ravis World Engine

## Current State
All planned features for the current development cycle have been implemented and verified. The engine is fully functional with 7 simulation stages and 5 map visualization layers.

## Recently Completed (This Session)
1. **Hash-Based 3D Noise** — Replaced separable trigonometric noise with a non-separable hash function (using irrational vector dot products) to permanently fix mirror symmetry artifacts across the globe. Added FBM support.
2. **14 New Geophysical Parameters** — Added UI and backend support for:
   - *Tectonics:* Crust Fraction, Plate Speed (cm/yr), Primary (Pangea) & Secondary (Americas) Clustering, Crust Warping.
   - *Geophysics:* African Superswells, Shallow/Deep Plumes (Hawaii/Iceland), Old Mountains/Hills (Appalachian/Caledonian), Stochastic Uplifts, and Volga Uplands.
   - *Paleoclimate:* LGM Temp Anomaly (Ice age temps) and Post-LGM Sea Level Rise (computes effective sea level).
3. **Hydrology BFS Rivers** — Eradicated mega-lakes by routing rivers out of endorheic basins through mountains (canyons) directly to the sea.
4. **Inverse Mapping & Tectonics** — Refactorized plate projection to "Inverse Mapping" (Backward Advection) to mathematically eliminate grid tearing artifacts using the inverse Rodrigues Rotation formula. Scaled down angular rotation velocities to prevent continent disintegration.
5. **Voronoi Biomes & Automata** — Resolved the mega-desert issue by seeding biomes via Holdridge, flood-filling the map, and applying 2 passes of cellular automata for organic clusters.
6. **Accurate Map Rendering** — Updated MapExporter to exactly match the ImGui legend color bands, rendering crisp elevation, and setting dry land to pure dark gray to highlight bright rivers/lakes.
7. **Oceanic Plates & Orogenesis** — Explicitly defined Oceanic vs Continental plates with negative crustal bases, driving trench formation on subduction and huge Himalayas-style ranges during continental collisions.
8. **Continuous Global FBM Noise** — Eradicated giant cliffs, "walls", and terrain tearing by removing per-plate noise offsets and abandoning the advective tectonic drift approach in favor of stable, seamless geometric Voronoi borders.
9. **Precipitation Baseline Rebalance** — Implemented a baseline evaporation/rain cycle for flat plains in the CUDA `AtmosphereSimulator`, curing the "megadesert" issue and restoring grassy plains and steppes.
10. **Steep Slope 3D Masking** — The engine dynamically calculates local vertex slopes in real-time, preventing snow from rendering on vertical cliffs to expose raw bedrock.

## Active Focus
- [ ] Connect the ocean simulator to the atmospheric climate (Wind-driven SWE / Munk vorticity).
- [ ] Implement geological and paleoclimatic constraints based on parameters (LGM temp anomalies, post-LGM sea level rise).


## Known Issues
- **Performance at high subdivisions**: Pixel→cell lookup is O(n²) — needs spatial indexing for level 7+.
- **Tectonic Drift Overlap**: Some artifacts might occasionally happen during extremely high-speed plate rotations without dynamic remeshing.

## Next Steps (Candidate Work)
- Add `Vector3` operator overloads for cleaner code
- Encapsulate main.cpp globals into `AppState` struct
- Spatial indexing (KD-tree) for MapExporter at high resolutions
- Consider 3D globe rendering (Phase 7+)
