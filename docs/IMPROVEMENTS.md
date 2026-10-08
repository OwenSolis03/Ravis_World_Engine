# Mejoras para Ravis World Engine

**Fuentes:** auditoría del código actual de Ravis (rama `fix/phase1-noise-moisture`) y análisis
del repositorio [Azgaar/Fantasy-Map-Generator](https://github.com/Azgaar/Fantasy-Map-Generator)
(FMG, licencia MIT — las *ideas* se pueden adoptar libremente; el código JS no aplica directo a C++/CUDA).

**Target de hardware:** laptop con **NVIDIA RTX 5060 Laptop (8 GB VRAM, Blackwell)** + CUDA.
Nada de MI210/HIP: la sección 4 adapta las reglas de HPC a GPU de consumo.

---

## Veredicto

Ravis ya tiene lo que FMG nunca tendrá: un pipeline causal (tectónica → clima → erosión →
hidrología → suelos → biomas) corriendo en GPU sobre una esfera real. FMG tiene lo que a Ravis
le falta por completo: **herramientas de edición, persistencia de proyecto y ergonomía de
cartógrafo**. La estrategia correcta no es copiar los editores de FMG uno a uno, sino tomar su
*catálogo de interacciones* y reimplementarlo sobre el diferenciador de Ravis: **cada edición se
propaga físicamente** (una cordillera pintada produce sombra orográfica real, no un bioma
repintado a mano).

Prioridad recomendada, en orden:

1. **P0 — Fundaciones de edición:** guardar/cargar proyecto, `EditOp` stack + `lock_mask`,
   refactor SoA con campos residentes en GPU. Sin esto, ningún pincel es posible.
2. **P1 — Física que no mienta:** unidades SI, ciclo mensual de T y P, humedad emergente
   (eliminar la fuente por bandas de latitud), determinismo en GPU.
3. **P2 — Herramientas estilo FMG:** set de pinceles, presets de arquetipo de mundo, ríos
   vectoriales, perfiles de elevación, submapas, exportación.

---

## 1. Estado actual del motor (lo observado en el código)

| Módulo | Archivo | Estado |
|---|---|---|
| Malla Goldberg | `src/GoldbergPolyhedron.cpp` | Funciona; niveles 3–8 (~640 a ~655K celdas) |
| Tectónica | `src/TectonicSimulator.cu` | Placas Voronoi deformado, subducción, plumas, edad de corteza |
| Atmósfera | `src/AtmosphereSimulator.cu` | T por latitud+lapse, SWE de viento en CUDA, advección de humedad en CPU |
| Erosión | `src/ErosionSimulator.cu` | 200K gotas en CUDA con `atomicAdd`, erosión costera |
| Hidrología | `src/HydrologySimulator.cpp` | Drenaje por descenso más empinado, lagos por BFS, efecto ribereño |
| Pedología | `src/PedologySimulator.cpp` | Roca madre + suelos por clima |
| UI | `src/main.cpp` | ImGui: sliders de parámetros + "Generate World" + globo 3D + 7 mapas 2D |

**Lo que no existe hoy:** guardar/cargar, undo, ningún pincel, ninguna edición del terreno,
ciclo estacional, unidades físicas coherentes. El único "editor" es regenerar el mundo entero
con otros parámetros. Ese es exactamente el modo de uso que FMG demuestra que no basta.

---

## 2. Qué tomar de Azgaar (y cómo traducirlo a Ravis)

FMG organiza su código en `generators/` (simulación), `controllers/` (~100 editores/overviews),
`renderers/` y `services/io/`. De ahí salen estas adopciones concretas:

### 2.1 Formato de proyecto con invariante de recarga — **la mejora más urgente**
FMG serializa el mundo entero a un único `.map` (JSON) con el invariante explícito: *guardar y
recargar debe reproducir el estado exacto*. Ravis no guarda nada: cerrar la app destruye el mundo.

**Para Ravis:** un `.ravis` (JSON o binario versionado) con `SimulationParameters`, semilla,
versión del motor, `edit_stack` y `lock_mask`. Los campos derivados (T, P, biomas) no se
serializan — se regeneran por determinismo — o se guardan solo como caché. Esto convierte el
determinismo que ya existe (misma semilla → mismo mundo) en persistencia casi gratis.

### 2.2 Pila de ediciones + pinceles (el corazón de "world edit tools")
El editor de heightmap de FMG (`heightmap-editor.ts`, 2.2K líneas) ofrece las herramientas
`Hill, Pit, Range, Trough, Strait, Mask, Invert, Add, Multiply, Smooth`, más un `vertex-brush`
y un `paint-editor`. Pero FMG muta las alturas **destructivamente** (un `Uint8Array` 0–100):
no hay historia, la resolución está fija, y rehacer el clima tras editar es un botón manual.

**Para Ravis, hacerlo mejor:** guardar **operaciones**, no deltas de celda:

```cpp
struct EditOp {          // serializable, independiente de la resolución
    EditOpType type;     // RAISE, LOWER, FLATTEN, SMOOTH, RIDGE_NOISE,
                         // MOUNTAIN_RANGE (polilínea), SET_LAND, SET_OCEAN, PAINT_LITHOLOGY
    Vector3   center;    // o polilínea de puntos sobre la esfera
    float     radius_m;  // radio geodésico, no euclidiano en lat/lon
    float     intensity; // Δh en metros
    Falloff   falloff;   // coseno / gaussiana
    uint64_t  seed;      // para los pinceles con ruido
};
// h_final = compose(h_generated, edit_stack);  h_generated es inmutable al editar
```

- La elevación final se compone: base generada + pila de ediciones. Undo = pop de la pila.
- `lock_mask`: las celdas editadas quedan fijadas; la erosión y la isostasia **no** las tocan
  (la edición del usuario manda). FMG no necesita esto porque nada corrige el terreno; Ravis sí.
- Las distancias del pincel se miden como geodésicas (`acos(dot(p, center))·R`), porque en los
  polos la distancia lat/lon deforma el pincel.
- El trazo `MOUNTAIN_RANGE` de FMG (polilínea con perfil transversal) es la herramienta más
  valiosa para fantasía: en Ravis debe producir sombra orográfica emergente al recalcular.

### 2.3 Presets de arquetipo de mundo (heightmap templates → presets tectónicos)
FMG arranca de plantillas con nombre: `volcano, highIsland, lowIsland, continents, archipelago,
atoll, mediterranean, peninsula, pangea, isthmus, shattered, taklamakan, oldWorld, fractious`.
Es ergonomía pura: el usuario elige una *intención* y luego edita.

**Para Ravis:** no pintar alturas, sino presets de `SimulationParameters`. Ya existen los knobs
(`num_plates`, `crust_fraction`, `primary_clustering`, `secondary_clustering`, plumas,
superswells): un preset "Pangea" = `primary_clustering≈0.9, num_plates≈7`; "Archipiélago" =
`crust_fraction≈0.12, shallow_plume_freq` alto; "Mediterráneo" = dos masas con mar interior vía
clustering secundario. Un combo en ImGui + una tabla de structs. Barato y de alto impacto.

### 2.4 Editores por capa con propagación (lo que FMG hace a mano, Ravis lo deriva)
FMG tiene editores de biomas, ríos, lagos, costa, hielo y relieve donde el usuario **pinta el
resultado**. En Ravis eso rompería la causalidad (principio 3 del motor). La traducción correcta:

| Editor FMG | Equivalente causal en Ravis |
|---|---|
| Biomes editor (pintar bioma) | No pintar biomas: editar T/P locales o litología, y el bioma se re-deriva |
| Rivers editor (dibujar río) | Editar elevación (valle) y dejar que el drenaje encuentre el cauce |
| Lakes editor | Pincel `LOWER` + recálculo de depresiones (ya existe la detección de lagos) |
| Coastline editor | `SET_LAND / SET_OCEAN` → invalida máscara tierra-mar → clima y evaporación |
| Ice editor | Derivado de T mensual (hoy ya se deriva de `temperature < 0.2`) |

Para que esto sea interactivo hace falta el **DAG de invalidación** y el **preview**:
- Cada campo declara sus dependencias; una edición marca sucios solo sus descendientes.
- Preview = malla gruesa (nivel 5–6), arranque en caliente desde la última solución, iteraciones
  limitadas, **el mismo código** que la simulación completa (si divergen, el preview miente).
- Regla crítica: la humedad se recalcula **global** (una cordillera cambia la lluvia miles de
  km a sotavento); la hidrología puede ser incremental por cuenca.
- La erosión **no** entra al DAG reactivo: es un paso de época explícito, o cada trazo del
  pincel "se derrite".

### 2.5 Diagnóstico y lectura del mundo (baratos, alto valor)
- **Cell inspector** (`cell-info.ts` de FMG): click en una celda → panel con elevación, T, P,
  bioma, roca, suelo, caudal, placa. Ravis ya tiene todos los datos en `Cell`; es solo UI de picking.
- **Elevation profile**: trazar una línea y ver el corte de elevación (y de P — ahí se *ve* la
  sombra orográfica, que además sirve como test visual de la física).
- **Temperature graph / charts**: curvas zonales de T y P por latitud. Con ellas, el test
  "desiertos en 15–35°, ecuador húmedo" se verifica de un vistazo.
- **Overviews tabulares** (rivers overview, features overview): top-N ríos por caudal, lagos
  por área. `logWorldStats()` ya calcula la mitad; falta mostrarlo en ImGui en vez de stdout.

### 2.6 Ríos vectoriales para el render
Ravis marca `river_flow` por celda y pinta la celda entera azul; FMG construye polilíneas con
ancho ~ caudal y meandros. Con `downstream_id` ya existe el árbol de drenaje: extraer polilíneas
(celda → celda aguas abajo), suavizarlas con Catmull-Rom sobre la esfera y renderizarlas como
líneas con grosor `∝ sqrt(Q)`. Mismo dato, mapa radicalmente más legible.

### 2.7 Submapa / re-muestreo regional
El `submap-tool` + `resample.ts` de FMG recortan una región y la regeneran a mayor densidad.
El equivalente Ravis: re-rasterizar `h_generated + edit_stack` sobre una malla localmente
refinada (o un parche plano proyectado) para la región que el usuario quiere detallar. Posible
precisamente porque las `EditOp` son independientes de la resolución (§2.2).

### 2.8 Lo que NO tomar (todavía)
Culturas, estados, religiones, burgos, rutas, economía (markets/goods/production), militar,
emblemas, nombres por cadenas de Markov. Es la mitad humana de FMG y es excelente, pero es otra
capa encima del motor físico (y en Ravis pertenece al dominio de worldbuilding narrativo, no al
engine). Lo único que vale la pena hoy: **exportar** los campos (elevación, ríos, biomas, costas)
a GeoJSON/PNG equirrectangular para que esa capa humana —propia o incluso el propio FMG— pueda
montarse encima después.

---

## 3. Deudas físicas del motor actual (impiden que la edición se propague bien)

En orden de severidad (síntoma visible primero):

1. **Humedad por bandas de latitud hardcodeadas.** `calculatePrimaryClimate()` asigna
   `moisture_base` por regla ("<30° → seco a 0.1, 30–60° → sube a 0.6…") y `simulateMoisture()`
   la re-inyecta cada iteración como término fuente. Es la regla ad hoc que el motor se prohíbe:
   los desiertos subtropicales deben *emerger* de la circulación (celdas de Hadley) y del
   transporte de humedad, no estar escritos. Síntoma: si el usuario edita un continente, la
   lluvia "recuerda" la banda terrestre aunque la geografía ya no la justifique; los monzones
   son imposibles. Arreglo: evaporación ∝ T del océano como única fuente, advección por el
   viento simulado, condensación por enfriamiento (orográfico + por latitud vía T). La banda
   seca de los 30° debe salir de la divergencia del viento, que el solver SWE ya produce.
2. **Una sola T anual, sin estaciones.** Köppen y los monzones necesitan los 12 meses de T y P
   (la insolación con inclinación axial ya está parametrizada a medias con el `axial_tilt`
   fantasma en `calculateTemperatures`). Síntoma: no hay biomas mediterráneos "de verdad"
   (lluvia invernal), ni sabana vs. selva por estacionalidad — hoy se distinguen solo por total anual.
3. **Unidades normalizadas 0..1 con conversiones mágicas dispersas.** `temperature*55-15` (°C)
   vive en `AtmosphereSimulator`, `main.cpp` y `PlanetData.h`; `precip*400` (cm/yr) en el
   clasificador; `*8000` (mm/yr) en el log. Tres escalas distintas para el mismo campo.
   Migrar a SI internamente (`temperature_K`, `precip_kg_m2_s`, como pide la convención del
   motor) y convertir solo en el render. Esto es prerrequisito para cualquier validación de
   conservación (ΣE ≈ ΣP) y para que el panel de diagnóstico (§2.5) diga números con sentido.
4. **No determinismo en GPU.** `erosion_drop_kernel` muta `elevation` con `atomicAdd` desde
   200K hilos: el orden de suma de floats cambia entre corridas → mismo seed ≠ mismo mundo.
   Eso rompe undo/redo y el preview reproducible. Arreglo: erosión *pull-based* por celda
   (stream-power / Braun–Willett de punto fijo: cada celda lee a sus vecinos del buffer viejo y
   se escribe solo a sí misma), que además es el patrón correcto de stencil para GPU. La
   `wind_velocity` final también se normaliza (`Math::normalize`), descartando la magnitud —
   la advección de humedad pierde información de intensidad del viento.
5. **CFL sin control en los solvers iterativos.** `dt=0.05` fijo en el SWE y advección de
   humedad "celda entera por paso": a resoluciones altas la advección viola CFL y produce los
   artefactos clásicos (humedad negativa / tablero de ajedrez — el suavizado posterior de
   precipitación es el parche que lo delata). Ligar `dt` al espaciamiento de la malla.
6. **Erosión que se comería las ediciones.** Hoy no hay ediciones, pero cuando las haya
   (P0), la erosión y la costera deben respetar `lock_mask` desde el día uno.

---

## 4. Plan CUDA para RTX 5060 Laptop (8 GB, Blackwell) — reemplaza la guía MI210/HIP

El objetivo HPC original (EPYC + MI210 + SLURM + HIP) no aplica. Reglas adaptadas:

### 4.1 Build
- `CMakeLists.txt` fija `CMAKE_CUDA_ARCHITECTURES 89` (Ada). La 5060 es **Blackwell,
  `sm_120`**: hoy corre vía JIT del PTX embebido, con costo de arranque y sin optimización
  nativa. Cambiar a `set(CMAKE_CUDA_ARCHITECTURES 89;120)` (requiere CUDA ≥ 12.8; si la
  toolchain es más vieja, dejar `89` y confiar en el JIT). No hace falta MPI ni HIP: capas
  fuera, un solo target.
- Warp = **32** (no 64 como CDNA2): bloques múltiplos de 32; los `blockSize = 256` actuales
  están bien.

### 4.2 Precisión: invertir la regla de la MI210
La MI210 tiene FP64 a ~1:1; la 5060 lo tiene a **1:64** — `double` en kernels es veneno.
- Campos de estado y kernels: `float`.
- Sumas de conservación y convergencia (ΣE, ΣP, balance energético): reducción en dos fases
  con orden fijo — sumas parciales por bloque en `float`, suma final **ordenada en `double` en
  CPU** (o Kahan en device). Nunca `atomicAdd` de float para estado ni para los totales.
- Geometría de la malla (generación del poliedro): `double` en CPU está bien, se hace una vez.

### 4.3 Memoria: 8 GB sobran; el ancho de banda no
Presupuesto real: nivel 8 ≈ 655K celdas. Con ~40 floats/celda (estado completo + 12 meses de
T/P) ≈ **105 MB**. Incluso nivel 10 (~10.5M celdas) cabe holgado. La restricción de la 5060 no
es VRAM sino **ancho de banda** (~GDDR7 de consumo vs 1.6 TB/s de HBM2e), y todos estos kernels
son memory-bound. Consecuencias:

- **SoA obligatorio.** `std::vector<Cell>` (AoS, con `std::vector<size_t> neighbors` por celda)
  se re-empaqueta hoy a arrays planos en *cada* llamada de *cada* módulo. Mover el mundo a un
  `WorldFields` SoA (`std::vector<float> elevation_m, temperature_K, ...` + CSR de vecinos
  rellenado a 6 con centinela) y que los módulos operen sobre él directamente. Es el mismo
  layout que ya usan los kernels — el refactor elimina el empaquetado, no lo crea.
- **Campos residentes en GPU.** Hoy cada módulo hace malloc → H2D → kernel → D2H → free
  (AtmosphereSimulator sube posiciones y vecinos, los baja; ErosionSimulator vuelve a subir
  exactamente lo mismo). Crear un `DeviceWorld` persistente que viva toda la sesión: se sube
  una vez tras la tectónica, los módulos encadenan kernels, y solo se baja lo que el render
  necesita. En el preview interactivo (pincel a <1 s) esto es la diferencia entre posible e imposible.
- Quitar los `cudaDeviceSynchronize()` entre kernels del mismo stream: el stream ya ordena;
  sincronizar solo antes de leer resultados. Con el bucle SWE de 100+ iteraciones, considerar
  **CUDA Graphs** para eliminar el overhead de lanzamiento (en laptop, con WDDM en Windows, el
  costo por lanzamiento es mayor que en Linux).
- **Interop CUDA–OpenGL** para el globo: registrar el VBO con `cudaGraphicsGLRegisterBuffer` y
  escribir colores/desplazamientos desde un kernel, en lugar del camino actual
  GPU → CPU (`cells`) → CPU recorre 655K celdas por frame de recoloreo → VBO.
- Ordenar las celdas por curva de llenado del espacio (Hilbert sobre la esfera) al generar la
  malla: mejora coalescencia en GPU y caché en CPU. Vale igual que en el plan MI210.

### 4.4 Qué va en GPU y qué no
- GPU: operadores de malla, EBM/temperatura mensual, SWE de viento, advección de humedad
  (hoy en CPU — es un stencil perfecto y es lo que el preview recalcula global en cada trazo),
  erosión pull-based, Köppen, render.
- CPU + OpenMP: Priority-Flood / resolución de depresiones y el enrutado de drenaje
  (secuenciales por naturaleza, baratos frente al clima; la acumulación de caudal puede ir a
  GPU como punto fijo si algún día domina el perfil). Regla: si es secuencial y barato, no
  forzarlo a GPU.
- En laptop, la GPU también dibuja el escritorio: dejar `swe_iterations` y el tamaño del
  preview configurables para no congelar la UI (o usar un stream de baja prioridad).

### 4.5 Validación con física (CI en la laptop)
- Tolerancia CPU↔GPU documentada y verificada en malla chica (p. ej. |ΔT| < 1e-3 K tras un año).
- Tests de regresión: Tierra (desiertos 15–35°, Amazonas/Congo húmedos, Atacama/Gobi/Tíbet en
  sombra), y cordillera N–S idealizada con vientos del oeste → barlovento húmedo / sotavento seco.
- Conservación: ΣEvaporación ≈ ΣPrecipitación en estado estacionario, en `double`.

---

## 5. Roadmap propuesto

### P0 — Fundaciones (sin esto no hay editor)
1. Refactor SoA (`WorldFields`) + `DeviceWorld` residente + quitar sync/roundtrips redundantes.
2. Formato de proyecto `.ravis` (params + seed + edit_stack + lock_mask) con invariante de recarga.
3. `EditOp` stack + composición `h_final = compose(h_generated, edit_stack)` + `lock_mask`.
4. Primer pincel (`RAISE/LOWER` geodésico) + undo por pop + recálculo completo al soltar
   (todavía sin DAG: recomputar clima→hidrología→biomas entero; en nivel 5–6 ya es interactivo).

### P1 — Física creíble y determinista
5. Unidades SI internas; conversión solo en I/O y render.
6. Erosión pull-based determinista (sin `atomicAdd` de estado); respeta `lock_mask`.
7. Humedad emergente: evaporación + advección + condensación; eliminar las bandas de latitud.
8. Ciclo mensual (12×T, 12×P) + clasificador Köppen; CFL ligado a resolución.
9. Suite de validación física (§4.5) corriendo en CI.

### P2 — Caja de herramientas estilo FMG
10. Set completo de pinceles: `FLATTEN, SMOOTH, RIDGE_NOISE, MOUNTAIN_RANGE` (polilínea),
    `SET_LAND/SET_OCEAN`, `PAINT_LITHOLOGY`; DAG de invalidación + preview con arranque en caliente.
11. Presets de arquetipo (Pangea, Archipiélago, Mediterráneo, Shattered…) como presets tectónicos.
12. Cell inspector, perfil de elevación, gráficas zonales de T/P, overviews de ríos/lagos.
13. Ríos vectoriales (polilíneas con grosor ∝ √Q) y render de costas suavizadas.
14. Exportación: PNG equirrectangular por capa, GeoJSON (costas/ríos/biomas), heightmap de 16 bits.
15. Submapa regional re-muestreando `h_generated + edit_stack` a mayor resolución.
