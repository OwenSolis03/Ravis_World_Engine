# Project Brief — Ravis World Engine

## Purpose
Ravis World Engine is a **procedural planetary simulation engine** that generates realistic Earth-like planets with physically-based geological, atmospheric, and hydrological systems. It runs as a desktop application with a Dear ImGui control panel for real-time parameter tuning.

## Core Goals
1. **Geological Realism**: Tectonic plates, subduction, orogenesis, isostasy, hotspot volcanism
2. **Granular Lithology**: 5 rock types (Basalt, Granite, Sandstone, Shale/Limestone, Metamorphic) with differential erosion
3. **Climate Simulation**: Latitude-based temperature, Coriolis-driven wind circulation, orographic precipitation
4. **Hydrological Systems**: Flow accumulation rivers, endorheic basin lakes, riparian moisture effects
5. **Interactive UI**: All parameters exposed via ImGui sliders with tooltips, configurable seed for deterministic/random worlds

## Target Users
- Worldbuilders, game designers, and procedural content creators
- Students and researchers interested in geophysics simulation

## Non-Goals (Current Scope)
- Not a game engine — no renderer, no player interaction
- Not HPC-optimized yet — single-threaded except erosion (OpenMP)
- No networking or multiplayer
