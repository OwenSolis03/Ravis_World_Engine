# Qué adoptar de Azgaar's Fantasy Map Generator en Ravis

Fecha: 2026-10-07 · Fuente analizada: [Azgaar/Fantasy-Map-Generator](https://github.com/Azgaar/Fantasy-Map-Generator) (rama principal, arquitectura "FMG 2.0": TypeScript, `src/generators/` ~28K líneas).
Complementa a `IMPROVEMENTS.md`; las referencias §N.N apuntan a ese documento.

**Veredicto en una línea:** no copiar el clima de FMG (es exactamente el modelo ad hoc por bandas de latitud que Ravis prohíbe); copiar su **hidrología de lagos**, su **vocabulario de edición de terreno**, su **pipeline declarado con locks y resample**, y —el premio gordo— toda su **capa de geografía humana**, que Ravis no tiene y que se deduce al 100 % de los campos físicos que Ravis ya calcula mejor.

**Nota de target (una sola vez):** el skill del proyecto apunta a HIP/ROCm en MI210 con ruta CPU primero; por decisión de Owen el target pasa a ser **CUDA en una laptop RTX 5060 (8 GB VRAM)**. La sección 6 redimensiona todo a ese hardware. La ruta CPU se conserva solo como oráculo de determinismo y tests, no como target de rendimiento.

---

## 1. Qué es FMG por dentro (lo relevante para Ravis)

- Mapa **plano 2D** sobre grafo de Voronoi (~10K celdas típicas), alturas en `Uint8Array` 0–100 con nivel del mar fijo en 20. Sin esfericidad, sin unidades físicas.
- **Pipeline de generación declarado** (`generation-pipeline.ts`): ~40 pasos con id (`heightmap → features → lakes → temperature → precipitation → rivers → biomes → suitability → cultures → burgs → states → routes → religions → provinces → markets → production → taxes → military → markers → zones → journeys`). Hay un segundo pipeline (`ErasePipeline`) que re-ejecuta el subconjunto post-heightmap cuando el usuario edita el terreno.
- Todo el estado del mundo es **un solo objeto serializable** (`pack` + `grid`) → guardar/cargar `.map` reproduce el mundo exacto.
- Las entidades tienen **flag `lock`**: un burg/estado/label bloqueado sobrevive a las regeneraciones.
- Editores por capa (heightmap brush, ríos, estados, culturas, burgs, rutas, labels, emblemas, zonas...) — es, con diferencia, el generador con más herramientas de edición del género. **Lo que le falta no son editores, es propagación de consecuencias**: al editar alturas, `ErasePipeline` re-genera culturas/estados/burgs *aleatoriamente de nuevo* (salvo locks); no hay invalidación selectiva ni identidad estable del contenido. Esa es exactamente la brecha que el DAG de Ravis (§4.3) debe explotar como diferenciador.

## 2. Lo que NO hay que copiar

- **Precipitación** (`precipitation-generator.ts`): tabla `LATITUDE_MODIFIER = [4,2,2,2,1,...]` por bandas de 5°, vientos que "entran por los lados del mapa" en 4 direcciones cardinales, humedad como contador 0–255. Es el patrón "si está a 30°, desierto" que el principio nº 3 de Ravis prohíbe. La advección emergente de Ravis (aun con sus bugs de §2.3–2.4) es la apuesta correcta.
- **Temperatura**: gradiente por latitud + lapse rate, una sola anual. Ravis ya la tiene igual y el objetivo es superarla con 12 meses (§2.1), no igualarla.
- **Alturas uint8 0–100**: pérdida de precisión inaceptable para isostasia/erosión en metros.
- Su limitación estructural (mapa plano con bordes) no existe en la malla de Goldberg: ninguna de sus soluciones de borde aplica.

## 3. Adoptar — nivel físico (hidrología, directamente sobre §2.7–2.8)

FMG resuelve bien, con 30 líneas por pieza, lo que el carving destructivo de Ravis rompe:

### 3.1 Lagos con balance de agua: abiertos vs cerrados (`lakes.ts`, `river-generator.ts`)
- `resolveDepressions()` **rellena depresiones sin excavar el terreno** (sube el agua, no baja la roca).
- Cada lago calcula `flux` (Σ precipitación de su cuenca/orilla) y `evaporation` con una **fórmula tipo Penman**: `E = ((700·(T + 0.006·h))/50 + 75)/(80 − T)` por celda de lago.
- Si `flux > evaporation` y el lago no está en depresión profunda → **lago abierto**: su `outlet` es la celda de orilla más baja y el excedente `flux − evaporation` sigue río abajo. Si no → **lago cerrado** (endorreico, salado).

**Adopción en Ravis:** sustituir el carving de `buildDrainageNetwork` por priority-flood + este balance, con T y P reales de la atmósfera y unidades SI (la fórmula de Penman completa ya usa °C y mm — encaja directo). Resultado visible: existen el Caspio y el Gran Lago Salado, los ríos atraviesan lagos correctamente, y nada pisa las ediciones del usuario.

### 3.2 Ríos como entidades con descarga, anchura y meandros (`river-generator.ts`)
FMG separa **física** (flux por celda, m³/s en la desembocadura, confluencias en `cells.conf`) de **cosmética** (`addMeandering` + `getWidth(getOffset(flux,...))` solo al trazar el path). Ravis hoy solo tiene `river_flow` crudo pintado por umbral de percentil (tres umbrales distintos, §2.9).

**Adopción:** promover los ríos a entidades (`River { mouth, source, cells[], discharge_m3_s, length_m }`) construidas desde el drenaje; convertir `precipitation` acumulada a m³/s reales (área de celda × precip); anchura y meandros **solo en el exportador/render**. Las confluencias dan deltas y estuarios gratis, y los nombres (§5) necesitan la entidad.

### 3.3 Humedal y el bonus ripario bien hecho (`biomes-generator.ts`)
FMG suma al cálculo de humedad del bioma `flux/10` si la celda tiene río y clasifica `wetland` cuando humedad/T/altura lo piden. Es exactamente el acoplamiento ripario→bioma que en Ravis es un no-op (§1.4): la corrección de §1.4 debe hacer esto — que la clasificación lea precipitación **+ contribución fluvial**, no un campo `moisture` que nadie consume.

## 4. Adoptar — arquitectura de edición (el punto fuerte a superar)

### 4.1 Pipeline declarado = el esqueleto del DAG de Ravis
FMG demuestra que basta una lista de pasos con id y un runner (`pipeline.ts`) para tener regeneración parcial utilizable. El DAG de Ravis (§4.3) es esto + aristas de dependencia explícitas + caché por nodo. **Recomendación:** implementar primero el pipeline declarado plano (una tarde de trabajo, ordena `runSimulation` ya), y añadir las aristas después. Cada paso: función pura campos→campos, como ya exige el skill.

### 4.2 El DSL de plantillas de heightmap = macros de pincel (`heightmap-templates.ts`)
FMG genera **todos** sus mundos ejecutando scripts de texto de ~10 líneas sobre 8 operaciones primitivas:

```
Hill 1 90-100 44-56 40-60      ← blob gaussiano con decaimiento aleatorio
Range 1.5 30-55 45-55 40-60    ← cordillera entre dos puntos con ruido de camino
Trough / Pit / Strait          ← valle, cráter, estrecho
Add / Multiply rango           ← modificar por franja de altura ("land", "all", "20-100")
Smooth / Mask / Invert         ← suavizar, atenuar bordes, simetrías
```

Esto es **el vocabulario del pincel de edición de Ravis**, ya probado con miles de usuarios: las mismas primitivas sirven como (a) brochazos interactivos y (b) "sellos" componibles (volcán, archipiélago, continentes...) que el usuario estampa. **Adopción:** implementar `Hill/Range/Trough/Pit/Smooth` como operaciones geodésicas (BFS por distancia esférica, ya hay código casi idéntico en `applyTerrainFeatures`, hoy muerto §1.2) que escriben en la **capa `user_edits`** — idealmente como deltas de espesor cortical para que la isostasia (§2.5) las respete en vez de pelearse con ellas — y un parser trivial del formato de texto para plantillas/sellos compartibles.

### 4.3 Locks por entidad y estado serializable
- El flag `lock` de FMG generaliza "la edición del usuario manda" a *todas* las capas: no solo terreno pintado, también "este bosque se queda", "esta ciudad no se mueve". En el DAG: un nodo con contenido bloqueado re-usa el resultado anterior en vez de recalcular, y la invalidación fluye alrededor.
- El invariante de FMG "guardar y cargar reproduce el mundo exacto" es la versión de producto del principio de determinismo nº 5. Ravis no tiene aún formato de guardado: definirlo pronto (campos + semilla + parámetros + lista de ediciones + locks) y hacer del *replay* de ediciones el mecanismo de undo/redo.

### 4.4 Resample (`resample.ts`) = el puente preview↔completa
FMG re-proyecta todos los campos y entidades al cambiar la densidad de malla o recortar una región (submapa), con quadtree del padre + suavizado. Es la pieza UX que el principio nº 4 de Ravis necesita: editar en nivel 6 (preview caliente) y transferir las ediciones a nivel 9–10 (simulación completa). En Goldberg es más fácil que en Voronoi: la jerarquía de subdivisión da el mapeo padre→hijo casi gratis (cada vértice de nivel n existe en nivel n+1). **Adopción:** guardar las ediciones en coordenadas esféricas continuas (no por id de celda) y rasterizarlas a cualquier nivel.

## 5. Adoptar — la capa de geografía humana (lo que Ravis no tiene en absoluto)

Es el 70 % del valor percibido de FMG y es **estrictamente aguas abajo** de los campos que Ravis ya calcula: ningún cambio de física, solo consumidores nuevos. En el DAG cuelgan al final, así que una edición de terreno re-deriva civilizaciones coherentemente — justo lo que FMG no puede hacer de forma estable. Orden de adopción por valor/esfuerzo:

1. **Suitability + población** (`population-generator.ts`, ~40 líneas): puntuación por celda = habitabilidad del bioma + bonus por flux fluvial normalizado + bonus costero/puerto natural. Con los biomas y ríos de Ravis sale gratis y es la base de todo lo demás.
2. **Nombres** (`names-generator.ts` + `name-bases.ts`): cadenas de Markov por sílabas sobre ~40 bases culturales de nombres. Autocontenido, ~300 líneas, portable a C++ en un día; sin nombres no hay mapa de fantasía.
3. **Culturas y estados** (`cultures-generator.ts`, `states-generator.ts`): semillas en celdas de alta puntuación + **expansión por coste** (Dijkstra multi-fuente donde cruzar montañas/desiertos/agua cuesta más, modulado por tipo de cultura: naval, nómada, montañesa...). Las fronteras emergen siguiendo la geografía — con la física de Ravis, mejor que en FMG.
4. **Burgs** (`burgs-generator.ts`): capitales = top suitability con espaciado mínimo (reintento con spacing reducido si no caben); ciudades menores por puntuación aleatorizada; detección de **puertos** (celda de agua adyacente navegable, "haven"); población derivada de suitability con multiplicadores capital/puerto.
5. **Rutas** (`routes-generator.ts`): A* sobre celdas con costes por terreno, ríos cruzables (×1.5) y mar por tramos (costa ×1 → océano lejano ×8), penalización de giros bruscos; conecta capitales→pueblos→puertos. En la malla de Goldberg es el mismo algoritmo.
6. Después, en este orden: **provincias**, **religiones**, **zonas** (desastres, claims), **markers** (POIs con lore), **military**, y la **economía nueva de FMG** (`goods/markets/production/taxes`: recursos por bioma/geología → mercados → producción → impuestos — encaja con la litología y pedología que Ravis ya tiene y FMG no).

Todo esto es trabajo de grafos sobre decenas de miles de celdas: **corre en CPU sin despeinarse**; no gastar VRAM ni kernels en ello.

## 6. Retarget: CUDA en laptop RTX 5060 (8 GB)

### 6.1 Build
- La 5060 es Blackwell, compute capability **12.0** → requiere CUDA Toolkit ≥ 12.8. En `CMakeLists.txt`: `set(CMAKE_CUDA_ARCHITECTURES "120")` (o `"89;120"` si la 4090 de desarrollo sigue en uso; mejor aún `native` como default de cache: `set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "...")`). El `89` hardcodeado actual genera PTX que la 5060 tendrá que JIT-compilar en el primer arranque, o directamente fallar según flags.
- Abandonar los planes HIP/MPI del skill; mantener la ruta CPU de referencia (§4.1 de IMPROVEMENTS) **solo** como oráculo: tests de determinismo comparan CPU vs GPU con tolerancia documentada, y es el fallback para quien no tenga NVIDIA.

### 6.2 Presupuesto de VRAM (8 GB) — qué resolución cabe
Con layout SoA (§4.2 de IMPROVEMENTS): ~40 campos float por celda + vecinos CSR (6 int + offset) ≈ **190–220 B/celda**. Clima mensual (12 × T, P, viento 3f) añade ~240 B/celda.

| Nivel | Celdas | Campos base | + clima mensual | ¿Cabe en 8 GB? |
|---|---|---|---|---|
| 7 | 163 842 | 0.03 GB | 0.07 GB | Sobra (preview) |
| 8 | 655 362 | 0.13 GB | 0.29 GB | Sobra |
| 9 | 2 621 442 | 0.52 GB | 1.1 GB | Sí, holgado |
| 10 | 10 485 762 | 2.1 GB | 4.6 GB | **Sí — target de simulación completa** |
| 11 | 41 943 042 | 8.4 GB | 18 GB | No (haría falta out-of-core/tiling) |

**Veredicto:** diseñar para **nivel 10 como simulación completa** (~49 km²/celda, celdas de ~7 km — más fino que FMG en órdenes de magnitud) y **nivel 6–7 como preview** interactivo. El TODO de `TectonicSimulator.cu:599` sobre nivel 13 / 510M celdas / multi-GPU queda fuera de alcance: borrarlo del horizonte simplifica todas las decisiones.

### 6.3 Lo que de verdad limita en una 5060 laptop
No es la VRAM, es el **ancho de banda** (~300–450 GB/s GDDR7 según TGP) y los viajes host↔device:
1. **Contexto GPU persistente** (§4.2 de IMPROVEMENTS): hoy cada módulo re-sube posiciones y vecinos en cada llamada. Subir la malla una vez por generación y dejar los campos residentes entre módulos ahorra más que cualquier optimización de kernel.
2. Quitar los `cudaDeviceSynchronize()` por iteración del solver de viento (dos por paso × hasta 5000 pasos): en un solo stream los kernels ya se ordenan solos; sincronizar solo al leer de vuelta.
3. Los kernels son memory-bound: el SoA nativo y vecinos CSR compactos (ya es el formato que los kernels consumen) importan más que los FLOPs.
4. **Determinismo en CUDA** (principio nº 5): los `atomicAdd` de la erosión por gotas dan resultados distintos por ejecución. Opciones, en orden de preferencia: (a) reorganizar la erosión a pasadas por celda (gather determinista, estilo ley de potencia de corriente de §2.8 — además es mejor física), o (b) mantener gotas pero acumular en buffers por bloque con reducción ordenada. Documentar la tolerancia CPU↔GPU restante.

## 7. Plan concreto (integrado con la hoja de ruta de IMPROVEMENTS.md)

| Fase | Qué | Origen |
|---|---|---|
| A (con los puntos 1–4 de IMPROVEMENTS) | Pipeline declarado de pasos con id + formato de guardado serializable | §4.1, §4.3 de este doc |
| B | Priority-flood + lagos abiertos/cerrados con Penman; ríos como entidades con descarga m³/s; anchura/meandros en render | §3.1–3.2 |
| C | Primitivas de pincel geodésicas (Hill/Range/Trough/Pit/Smooth) sobre capa `user_edits` + parser de plantillas/sellos; locks por entidad | §4.2–4.3 |
| D | Resample esférico preview↔completa (ediciones en coordenadas continuas) | §4.4 |
| E | Suitability+población → nombres Markov → culturas/estados por expansión de coste → burgs+puertos → rutas A* | §5 (CPU) |
| F | Provincias, religiones, zonas, markers, economía (goods/markets/production con la litología de Ravis) | §5 |
| Transversal | CMake a arch 120/native, contexto GPU persistente, erosión determinista, presupuesto nivel 10 | §6 |

**La tesis competitiva, explícita:** FMG tiene los mejores editores del género pero cada edición re-tira los dados del mundo humano; Ravis tiene física real y determinismo. Adoptando de FMG la hidrología de lagos, el vocabulario de edición y la capa humana, y colgándolo todo del DAG determinista, Ravis ofrece lo que FMG no puede: *pinta una cordillera y mira cómo el río se reencauza, el desierto de sombra aparece, la ruta comercial se desvía y la ciudad del paso de montaña pierde la mitad de su población — sin que nada más cambie.*
