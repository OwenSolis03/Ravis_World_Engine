# Ravis World Engine — Auditoría y mejoras propuestas

Fecha: 2026-10-07 · Alcance: todo el código en `src/` e `include/` (≈4.300 líneas C++/CUDA).
Prioridad de la auditoría: (1) física incorrecta que produce mundos plausibles pero equivocados, (2) violaciones de conservación, (3) no determinismo, (4) errores de malla, (5) rendimiento.

**Veredicto general:** la base es sólida (malla icosférica correcta con pentágonos manejados, pipeline tectónica→clima→erosión→hidrología→suelos bien ordenado, GPU funcional), pero hay un grupo de **acoplamientos rotos y código muerto** que hace que varios sliders de la UI y pasos enteros de la simulación no tengan ningún efecto, y la física de clima todavía es demasiado ad hoc para que los patrones realistas (desiertos subtropicales, sombra orográfica, monzones) *emerjan* en vez de pintarse por latitud.

---

## 1. Acoplamientos rotos y código muerto (arreglar primero: máximo efecto por línea cambiada)

### 1.1 `TectonicSimulator::simulate()` ignora `iterations` — no existe el bucle temporal
`src/TectonicSimulator.cu:964` recibe `iterations` (derivado de `planet_age_Myr / 5`) y nunca lo usa. Los kernels `tectonic_drift_kernel` (línea 256) y `tectonic_apply_advection_kernel` (línea 403) — que implementan la deriva de placas, la subducción por el "atacante", la creación de corteza en dorsales y la subsidencia térmica — **están escritos pero jamás se lanzan**. La simulación hace una sola pasada estática: estrés → difusión → uplift → hotspots → ruido FBM → suavizado.

**Síntoma visible:** el slider *Planet Age* solo cambia el número de gotas de erosión; *Subsidence Rate* no hace absolutamente nada (la subsidencia solo existe dentro del kernel de advección muerto); no hay gradiente de edad de corteza desde las dorsales (la edad se asigna uniforme aleatoria 10–200 Ma en `generatePlates`), así que la regla "la corteza más vieja subduce" no puede operar nunca.

**Recomendación:** activar el bucle temporal con los kernels de advección/deriva ya escritos, inicializar `crustal_age` en 0 en las dorsales y dejar que el gradiente emerja, y eliminar los buffers `d_next_age/d_next_thick/d_next_vx/vy/vz` si se decide no hacerlo (hoy se asignan y liberan sin usarse jamás: `src/TectonicSimulator.cu:1028-1035`).

### 1.2 `applyTerrainFeatures()` nunca se llama — 6 sliders de Geofísica muertos
`src/TectonicSimulator.cu:824` implementa superswells, montañas antiguas, colinas, uplifts y uplands, pero ninguna ruta de código la invoca. Los sliders *Superswells, Old Mountains, Old Hills, Small Uplifts, Uplands* de la UI no tienen ningún efecto (solo *Shallow/Deep Plumes* funcionan, vía `generateHotspots`).

**Recomendación:** llamarla desde `simulate()` después del uplift tectónico y antes del FBM, o borrarla junto con sus sliders. Lo peor es el estado actual: controles que mienten.

### 1.3 `OceanSimulator` es un stub que dice "complete"
`src/OceanSimulator.cu:80-89`: `simulateCurrents` imprime "Simulating Ocean Currents (Munk Model)... complete" y no hace nada; `ocean_munk_kernel` nunca se lanza. Las corrientes oceánicas no influyen en nada del clima.

**Síntoma visible:** no hay moderación costera tipo corriente del Golfo; las costas occidentales y orientales de los continentes a la misma latitud tienen el mismo clima, lo cual está mal (Europa vs. Labrador).

**Recomendación:** o conectar el solver de Munk y acoplar su salida a la temperatura superficial del mar antes de `calculateFullClimate`, o quitar la llamada de `main.cpp:113-114` y el log engañoso hasta que exista.

### 1.4 El re-asignado de biomas tras hidrología no hace nada (acoplamiento ripario roto)
`src/main.cpp:121-122` re-ejecuta `atmosphere.assignBiomes(params)` "porque la humedad riparia cambia los biomas". Pero `applyRiparianEffect` (`src/HydrologySimulator.cpp:229`) solo sube `cell.moisture`, y `assignBiomes` (`src/AtmosphereSimulator.cu:425`) clasifica usando **`cell.precipitation`**, que no se toca. La segunda llamada produce bit a bit el mismo resultado; el efecto ripario completo es hoy un no-op (nada aguas abajo lee `moisture` después de la hidrología).

**Síntoma visible:** no hay corredores verdes a lo largo de los grandes ríos en desiertos (no puede existir un Nilo).

**Recomendación:** decidir la semántica de `moisture` vs `precipitation` (ver §3.4) y hacer que el boost ripario alimente el campo que los biomas y suelos realmente leen.

### 1.5 `cell.biome` se calcula y no se renderiza nunca
`assignBiomes` hace una clasificación estilo Holdridge con PET (razonable), pero ni el exportador 2D (`src/MapExporter.cpp:43-70`, tabla Whittaker de 13 puntos por vecino más cercano sobre t/p crudos) ni el globo 3D (`src/main.cpp:640-655`, otra tabla distinta) leen `cell.biome`. El enum `BiomeType` solo alimenta a `PedologySimulator`. Además la leyenda de la UI (`src/main.cpp:419-424`) lista 15 nombres que no corresponden ni al enum (14) ni a las tablas de colores (13).

**Síntoma visible:** cualquier ajuste a la clasificación de biomas no cambia nada en pantalla; el mapa, el globo y la leyenda muestran tres clasificaciones distintas.

**Recomendación:** una sola fuente de verdad: renderizar `cell.biome` con una tabla `BiomeType → color` compartida entre exportador, globo y leyenda.

### 1.6 OpenMP se enlaza como REQUIRED y no se usa en ninguna parte
`CMakeLists.txt:33` exige OpenMP, pero no hay un solo `#pragma omp` en el proyecto. Toda la ruta CPU (advección de humedad, hidrología, suelos, carving) es mono-hilo.

**Recomendación:** paralelizar al menos `simulateMoisture` (bucle por celdas con buffers dobles, trivialmente paralelo) y los exportadores, o quitar el REQUIRED.

### 1.7 Fugas y restos en GPU
- `d_plate_centers_x/y/z` nunca se liberan (`src/TectonicSimulator.cu:1185-1195` libera ejes y velocidades pero no centros) → fuga por cada "Generate World".
- Los buffers `d_next_*` del kernel de advección muerto (§1.1) se asignan/liberan sin uso.
- `MapExporter::exportAllMaps` (PNG a disco) no es alcanzable desde la UI — o exponer un botón "Export PNGs" o borrarla.

---

## 2. Física incorrecta o faltante (lo que impide que los patrones emerjan)

### 2.1 Una sola temperatura y precipitación anual — Köppen y los monzones son imposibles
Todo el clima es un único estado estacionario: `temperature` y `precipitation` escalares por celda, normalizados 0–1. Sin ciclo mensual no puede haber clima mediterráneo real (lluvia de invierno), monzones (inversión estacional del viento), ni distinción continental/oceánica por amplitud térmica. El bioma "MEDITERRANEAN" actual se adivina desde la razón PET anual, que no puede distinguirlo de una estepa húmeda.

**Recomendación (la mejora de física de mayor valor):** calcular 12 estados mensuales variando la declinación solar, con los *mismos* kernels, y clasificar con Köppen sobre los ciclos de T y P. El costo es ~12× el paso de clima, que hoy es barato.

### 2.2 Unidades: casi todo es adimensional con constantes mágicas terrestres
- Temperatura 0–1 con conversión escondida `t*55 − 15 °C` (`AtmosphereSimulator.cu:423`), precipitación 0–1 con `*8000 mm` (línea 425) y una normalización arbitraria `/3` (línea 408).
- Coriolis "tweaked": `f = 2 · 1.5 · sin(lat)` (`AtmosphereSimulator.cu:93`) — Ω inventada para que se vean bandas.
- Inclinación axial 23° hardcodeada (línea 207), lapse rate `0.00012` por metro en unidades normalizadas (línea 224), gravedad 9.8 dentro del kernel de erosión, radio del planeta = 1 implícito en todas las distancias.

**Síntoma visible:** Ravis solo puede ser la Tierra; ningún parámetro planetario (radio, rotación, inclinación, constante solar) es editable, y los cambios de resolución alteran la física porque las distancias entre celdas cambian de magnitud pero las constantes no.

**Recomendación:** introducir un struct `PlanetPhysicalParams { radius_m, gravity_m_s2, rotation_period_s, axial_tilt_rad, solar_constant_W_m2, ... }`, pasar a unidades SI internas (K, kg/m²/s, Pa) y convertir solo en la E/S. Documentar la unidad en el nombre de cada campo (`elevation_m`, `precip_kg_m2_s`).

### 2.3 El viento descarta su magnitud — la fuerza del viento nunca importa
Tras el solver SWE, `calculateWinds` normaliza el vector (`AtmosphereSimulator.cu:316`), y la advección de humedad usa solo la dirección. Un vendaval y una brisa transportan la misma humedad. Además la presión inicial es solo `1 − temperature` (sin efecto orográfico sobre la presión) y el paso `dt = 0.05` es fijo sin comprobación CFL.

**Recomendación:** conservar la magnitud y usarla como fracción advectada por paso (con límite CFL explícito: fracción ≤ 1), y añadir una capacidad de saturación dependiente de la temperatura (Clausius-Clapeyron simplificada). Esto hace emerger desiertos subtropicales más secos y lluvias tropicales más intensas sin reglas por latitud.

### 2.4 La humedad no se conserva — ΣE ≠ ΣP por construcción
En `simulateMoisture`: los océanos inyectan `temperature * 0.1` por iteración sin límite, la lluvia que cae no vuelve a ningún reservorio, el clamp `min(1, moisture)` destruye masa, y la precipitación acumulada se renormaliza con el `/3` mágico. No hay forma de escribir el test de cordura ΣE ≈ ΣP.

**Recomendación:** tratar la humedad como columna de agua (kg/m²) con evaporación = f(T, océano), advección conservativa y precipitación como sumidero explícito; verificar el balance global en un test automático.

### 2.5 Isostasia al revés y clamps que matan las plataformas continentales
- `crustal_thickness` se **deriva de** la elevación (`generatePlates`, `TectonicSimulator.cu:804`; `bfs_uplift`, línea 855) en vez de que la elevación salga del espesor y densidad de la corteza. Es exactamente la trampa señalada en la revisión anterior del proyecto.
- `tectonic_fbm_kernel` fuerza continental ≥ +50 m y oceánico ≤ −100 m (`TectonicSimulator.cu:578,586`): la línea de costa coincide 1:1 con el tipo de placa.

**Síntoma visible:** no existen plataformas continentales someras, mares epicontinentales (Báltico, Hudson), ni continentes parcialmente inundados; el slider *Sea Level* (±500 m) apenas mueve las costas porque hay un escalón de 150 m vacío alrededor del 0.

**Recomendación:** elevación = compensación isostática de (espesor, densidad, edad) + componente dinámica; quitar los clamps y dejar que el nivel del mar corte donde corte.

### 2.6 Subducción decidida solo por `is_oceanic` local, sin polaridad
`tectonic_apply_stress_kernel` (`TectonicSimulator.cu:334`): en convergencia, si la celda es oceánica → fosa, si no → montaña. No hay distinción océano-continente vs continente-continente ni flotabilidad por edad/densidad (que además no puede funcionar sin el gradiente de edad, §1.1). Las fosas no forman pares fosa + arco volcánico asimétricos.

**Recomendación:** decidir la polaridad por flotabilidad (edad/espesor de las dos cortezas enfrentadas) y aplicar fosa en el lado que subduce y arco/cordillera en el que cabalga.

### 2.7 El carving hidrológico destruye cuencas endorreicas (y destruirá las ediciones del usuario)
`buildDrainageNetwork` (`HydrologySimulator.cpp:48-108`) hace BFS desde cada pit y **excava el terreno** (baja elevaciones hasta 1500 m de pared) para forzar que todo drene al mar. Consecuencias: casi no quedan pits → casi no hay lagos (`detectLakes` depende de pits), no pueden existir el Caspio, el Gran Lago Salado ni el Tarim; y cuando exista la capa de edición, este paso pisará las montañas pintadas por el usuario.

**Recomendación:** sustituir por *priority-flood* (Barnes et al.): llena depresiones con un nivel de agua hasta su punto de desborde sin tocar el terreno, define el outlet de cada lago y deja las cuencas endorreicas donde la evaporación supere la entrada. Es además O(n log n) global en vez del O(n²) actual (un BFS con vectores `visited`/`parent` de tamaño n **por cada pit**).

### 2.8 Erosión: constantes con unidades rotas y sin acoplamiento a la lluvia real
`erosion_drop_kernel` (`ErosionSimulator.cu:41-47`): `min_slope = 100.0f` como suelo de la pendiente hace que la capacidad sea siempre enorme e independiente del terreno real (la pendiente casi nunca supera 100 en estas unidades, así que el término de pendiente es constante); `velocity = sqrt(v² + elevDiff·9.8)` mezcla unidades; la erosión corre **antes** del clima completo (usa la humedad primaria por latitud) y una sola vez, así que la lluvia orográfica real nunca esculpe el terreno.

**Recomendación:** erosión basada en el flujo acumulado de la hidrología (ley de potencia de corriente, `E ∝ A^m · S^n`) tras el clima completo, con al menos un ciclo clima→erosión→clima; dimensionar las constantes con las distancias reales de celda.

### 2.9 Menores de física
- `detectLakes`: nivel de agua fijo pit+50 m y `river_flow` compartido por copia; sin balance entrada/evaporación.
- Comentario "top 5%" con código al 15% en `applyRiparianEffect` (`HydrologySimulator.cpp:204`); tres umbrales de río distintos (85 %, 90 %, 95 %) en hidrología, mapa hidro y mapa de biomas — unificar en una definición de "río".
- `coastal_erosion_kernel`: un solo tick con 5 m mágicos; depende del viento, que a esa altura del pipeline ya existe, pero su efecto es cosmético.
- Hielo = `temperature < 0.2f` repetido en 5 sitios (exportador, globo, leyenda) — constante sin nombre y sin conexión con 0 °C.

---

## 3. Determinismo y reproducibilidad (principio nº 5 del proyecto)

### 3.1 La semilla no alimenta el ruido: todos los mundos comparten el mismo campo FBM
`hash_noise3d`/`cpu_hash_noise3d` dependen solo de la posición. Con semillas distintas cambian placas y hotspots, pero el patrón de rugosidad continental, las isotermas onduladas y la distorsión de lluvia son **idénticos en todos los mundos**.

**Recomendación:** mezclar `params.seed` en el hash (offset del dominio o constante de mezcla).

### 3.2 Hash basado en `sin()` de argumentos enormes — frágil entre CPU y GPU
`sinf(x * 127.1f + ...)` con `* 43758.5453f` depende de la reducción de argumento de cada implementación de `sin`; CPU (glibc) y GPU (CUDA fast math) difieren, y el mismo mundo no es reproducible entre la ruta CPU futura y la GPU. Sustituir por un hash entero (PCG/xxhash sobre celda cuantizada) con gradientes tabulados.

### 3.3 Orden de vecinos no determinista
`extractCells` (`GoldbergPolyhedron.cpp:137-150`) construye los vecinos con `std::unordered_set`, cuyo orden de iteración es de implementación. Las sumas en punto flotante (advección, difusión de estrés, suavizados) dependen del orden → el mismo binario en otra stdlib/plataforma da mundos distintos bit a bit.

**Recomendación:** ordenar los vecinos canónicamente (por id, o mejor: en orden angular alrededor de la celda, que además hace posibles operadores de gradiente/divergencia bien definidos).

### 3.4 Otras fuentes de no determinismo y semántica confusa
- `atomicAdd` de floats en el kernel de erosión: el orden de las gotas varía entre ejecuciones → terrenos distintos con la misma semilla incluso en la misma GPU. Aceptable si se documenta una tolerancia; inaceptable como está, sin documentar.
- `moisture` vs `precipitation`: dos campos con significados solapados que distintos módulos leen a medias (la erosión lee `moisture` primaria; biomas y suelos leen `precipitation`; lo ripario escribe `moisture`). Definir: `moisture` = agua en columna atmosférica (estado transitorio), `precipitation` = flujo que llega al suelo (salida), y que todos los consumidores lean el segundo.

---

## 4. Arquitectura y rendimiento (ruta HPC)

### 4.1 CUDA obligatorio con arch 89 — contradice el objetivo (laptop primero, MI210 después)
`CMakeLists.txt:8-11`: `enable_language(CUDA)` incondicional y `CMAKE_CUDA_ARCHITECTURES 89` (RTX 40xx) hardcodeado. El proyecto no compila en la laptop sin NVIDIA ni en Yuca (MI210 es ROCm/HIP, no CUDA).

**Recomendación:**
1. Implementación CPU de referencia de cada kernel (son todos bucles por celda, portables en horas), detrás de `option(RAVIS_WITH_GPU ...)`.
2. Backend GPU portable: HIP directo (hipify-perl convierte este CUDA casi 1:1) o un wrapper fino de macros; arch como variable de cache, no constante.
3. El mismo código de física para preview y simulación completa, parametrizado (resolución, iteraciones), nunca dos implementaciones — hoy no hay preview, pero este es el momento de no bifurcar.

### 4.2 Layout AoS: `std::vector<Cell>` con 20 campos y un `std::vector` interno
Cada módulo GPU re-aplana `cells` a SoA y lo re-sube **en cada llamada** (tectónica, atmósfera, erosión repiten ~10 memcpys y mallocs de los mismos posiciones/vecinos). Además `Cell::neighbors` como `std::vector<size_t>` dispersa la memoria y mata el prefetch en la ruta CPU.

**Recomendación:** hacer de SoA el formato nativo (`PlanetFields { std::vector<float> elevation_m; ... }` + CSR plano para vecinos `neighbor_index[cell*6+k]` con `neighbor_count[cell]`, que ya es exactamente lo que todos los kernels consumen), y un contexto GPU persistente que suba la malla una vez por generación. `Cell` puede quedar como vista de conveniencia o eliminarse.

### 4.3 No existen el DAG de dependencias, la capa de edición ni el undo
Los tres pilares del producto ("la edición del usuario manda", invalidación selectiva, preview interactivo) no tienen ni esqueleto: `runSimulation` es un monolito que regenera todo desde cero. Cuanto más crezca el pipeline actual, más caro será introducirlos.

**Recomendación:** antes de añadir más física, extraer el grafo explícito de pasos (nodo = función pura campos→campos, aristas = dependencias declaradas) y una capa `user_edits` autoritativa que se re-aplica tras cada nodo que toque `elevation`. El carving hidrológico (§2.7) y la erosión son los primeros que deben respetarla.

### 4.4 Rendimiento puntual
- `buildPixelToCellMap`: búsqueda lineal O(píxeles × celdas) en GPU (`MapExporterCUDA.cu:45`). Con nivel 8 (655K celdas) y 1024×512 son ~3.4×10¹¹ comparaciones por generación. Un grid espacial o aprovechar que la malla es casi regular (celda ≈ f(lat,lon)) lo baja a O(píxeles).
- `tectonic_drift_kernel` usa `hash_noise3d(px + seed_offset, py − seed_offset, pz * seed_offset)` — con `seed_offset = 0` el tercer argumento colapsa a 0 (cuando se active §1.1, revisar).
- Doble `cudaDeviceSynchronize` por iteración del solver SWE: innecesario dentro de un mismo stream; con 5000 iteraciones el overhead de sincronización domina.

---

## 5. UI, hilos y render

### 5.1 Data race en los flags de 3D
`update_3d_geometry`, `update_3d_colors` y `active_planet` se escriben desde el hilo de simulación bajo `pixel_mutex` (`main.cpp:145-158`) pero el hilo de render lee los dos flags **sin** tomar el mutex (`main.cpp:522,543`). Son `bool` planos: UB formal y visibilidad no garantizada. Hacerlos `std::atomic<bool>` (como ya lo son `is_simulating`/`new_map_ready`) o leerlos bajo el lock.

### 5.2 Errores de CUDA matan el proceso desde un hilo secundario
Todos los `CHECK_CUDA` hacen `exit(EXIT_FAILURE)`; en el hilo de fondo eso derriba la app sin mensaje en la UI, y `is_simulating` quedaría colgado en diseños futuros. Propagar error (excepción capturada en `runSimulation` → estado "failed" visible en la UI).

### 5.3 OpenGL de función fija en un contexto 3.0
El globo usa `glMatrixMode/glLoadMatrixf/glVertexPointer` (compatibility profile) mientras ImGui usa GLSL 130. Funciona en drivers NVIDIA/Mesa de escritorio, pero no en core profile ni macOS. Migrar el globo a un VBO+shader minimal cuando se toque el render.

### 5.4 Detalles
- Leyendas desincronizadas de los datos reales (§1.5); la leyenda de pedología lista órdenes USDA (Aridisol, Mollisol, Oxisol, Gelisol, Inceptisol) pero el modelo solo tiene SAND/CLAY/LOAM/NONE.
- `glTexImage2D` re-crea la textura en cada cambio de capa; `glTexSubImage2D` basta.
- El combo dice "CUDA Iterations" — nombrarlo por lo que es físicamente (iteraciones del solver de viento).
- `PairHash` (`GoldbergPolyhedron.h:13`): `<< 32` es UB si `size_t` es de 32 bits; usar un mezclador estándar (boost::hash_combine / splitmix64).

---

## 6. Calidad de ingeniería (hoy: cero red de seguridad)

1. **Sin tests.** Mínimos a añadir, en orden de valor:
   - *Malla:* exactamente 12 celdas con 5 vecinos, el resto con 6; suma de ángulos/áreas ≈ 4π.
   - *Conservación:* ΣE ≈ ΣP en estado estacionario (requiere §2.4); energía absorbida ≈ emitida cuando haya balance energético.
   - *Regresión Tierra:* con topografía terrestre importada, desiertos en ~15–35°, Amazonas/Congo húmedos, Atacama/Gobi/Tíbet en sombra orográfica.
   - *Cordillera idealizada N–S* con vientos del oeste → barlovento húmedo, sotavento seco.
   - *Determinismo:* dos ejecuciones con la misma semilla → hash idéntico de todos los campos (CPU); tolerancia documentada en GPU.
2. **Sin README** (cómo compilar, requisitos de GPU, capturas), **sin LICENSE**, **sin CI** (un workflow que compile la ruta CPU cuando exista, y corra los tests, atraparía la mitad de los hallazgos de §1 automáticamente).
3. Tres copias del mismo hash noise (device y host en `TectonicSimulator.cu`, host en `AtmosphereSimulator.cu`) → un header `Noise.h` único compartido CPU/GPU, que además resuelve §3.2.
4. `seed` es `int` truncado desde el reloj; usar `uint64_t` de extremo a extremo.

---

## 7. Inconsistencia transversal del nivel del mar (merece sección propia)

`effective_sea_level() = sea_level + post_lgm_sea_rise` se usa en atmósfera, biomas y todos los renderizadores, pero **Hydrology, Erosion y Pedology usan `params.sea_level` crudo** (`HydrologySimulator.cpp:31,49,125,...`, `ErosionSimulator.cu:247`, `PedologySimulator.cpp:13,21`).

**Síntoma visible:** con *Post-LGM Sea Rise* > 0, hay ríos que fluyen y suelos que se forman en celdas que se pintan como océano, lagos "submarinos", y la erosión trata la plataforma inundada como tierra firme.

**Recomendación:** una única función (o campo precalculado `is_ocean`) consumida por todos los módulos. Es un cambio de una línea por módulo y elimina una clase entera de rarezas.

---

## Hoja de ruta sugerida (orden concreto)

| # | Tarea | Secciones | Esfuerzo | Impacto |
|---|-------|-----------|----------|---------|
| 1 | Unificar nivel del mar efectivo en todos los módulos | §7 | Bajo | Alto |
| 2 | Conectar o eliminar el código muerto: `applyTerrainFeatures`, bucle temporal tectónico, stub oceánico, fuga de `d_plate_centers` | §1.1–1.3, 1.7 | Bajo–Medio | Alto |
| 3 | Arreglar el acoplamiento ripario→biomas y renderizar `cell.biome` con leyenda única | §1.4–1.5 | Bajo | Alto |
| 4 | Semilla en el ruido + vecinos con orden canónico + hash entero | §3.1–3.3 | Bajo | Alto (determinismo) |
| 5 | Ruta CPU de referencia + CMake con GPU opcional (prepara HIP/MI210) | §4.1 | Medio | Alto |
| 6 | Tests mínimos + CI + README | §6 | Medio | Alto |
| 7 | Struct de parámetros planetarios + unidades SI | §2.2 | Medio | Medio–Alto |
| 8 | Clima mensual (12 estados) + Köppen + capacidad por temperatura + magnitud del viento | §2.1, 2.3 | Alto | Muy alto |
| 9 | Priority-flood en vez de carving destructivo; erosión por ley de potencia tras el clima | §2.7, 2.8 | Medio | Alto |
| 10 | Isostasia directa (elevación desde espesor/densidad) y subducción por flotabilidad | §2.5, 2.6 | Alto | Alto |
| 11 | SoA nativo + contexto GPU persistente | §4.2 | Medio | Medio (crítico antes de nivel 8+) |
| 12 | DAG de dependencias + capa de ediciones de usuario + preview | §4.3 | Alto | Es el producto |

Los puntos 1–4 son casi todo cableado, no física nueva: con ellos, lo que ya está escrito empezaría a comportarse como dice que se comporta.
