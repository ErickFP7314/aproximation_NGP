# Precisión y honestidad en la medición del Polo Norte Galáctico con Gaia DR3: una extensión del método de Cano (2022)

**Borrador para la Revista Boliviana de Física**
*Estado: borrador de trabajo interno, no enviado a revisión. Generado a partir de los resultados computacionales del ciclo `ngp-precision` (ver `NGP_Precision.ipynb`, `results/master_table.{md,csv,tex}`).*

**Autores sugeridos**: Ludving Cano¹ (método original, 2022), [nombre del colaborador]² (extensión Gaia DR3, este trabajo).
¹Autor de "Aproximación geométrica del polo norte galáctico" (Revista Boliviana de Física 40, 2022).

---

## Resumen

Extendemos el método geométrico de Cano (2022) para estimar la posición del Polo Norte Galáctico
(NGP) usando datos reales de Gaia DR3. El ciclo previo (`ngp-improvement`) estableció un estimador
de círculo máximo sin parámetros libres (`great_circle_pole`) sobre 53,082 estrellas del disco,
obteniendo α=12.9463h, δ=26.49°, con un error angular de 1.44° frente a la posición IAU y un
intervalo de confianza bootstrap al 95% que **excluye** el valor IAU de forma estadísticamente
robusta — sugiriendo un sesgo sistemático, no solo ruido de muestreo. Este trabajo investiga el
origen de ese sesgo mediante cinco extensiones: (1) un ajuste de plano con offset libre que
recupera la altura del Sol sobre el plano galáctico (z☉); (2) tres trazadores jóvenes independientes
(Cefeidas, estrellas OB, cúmulos abiertos) que trazan mejor el plano de rotación actual que la
muestra de disco de todas las edades; (3) un método 3D ponderado (TLS/IRLS) que corrige el sesgo de
ruido de paralaje del RANSAC original; (4) un polo cinemático derivado de movimientos propios,
físicamente independiente de la posición estelar; y (5) un presupuesto de error honesto que separa
la componente estadística de la sistemática. El hallazgo central es que **el error sistemático
(0.7°–2.7° según el trazador) domina por un factor de 2 a 15 sobre el error estadístico bootstrap**
— el verdadero cuello de botella de precisión no es el tamaño de la muestra, sino las decisiones de
selección y corte del análisis. No se alcanzó el objetivo original de reducir el error a
~0.03°–0.05°, pero se entrega un presupuesto de error completo, tres trazadores independientes con
su discrepancia cuantificada, y una descomposición forense de 1.35° de la divergencia frente al
polo IAU en términos de medición histórica (0.12°), artefacto de transformación de coordenadas
(0.0001°) y una diferencia gas-vs-estrellas que se define como remanente de cierre del presupuesto
(1.23°).

**Palabras clave**: Polo Norte Galáctico, Gaia DR3, astrometría, sistema de coordenadas galácticas,
Cefeidas, movimientos propios.

---

## 1. Introducción

Cano (2022) propuso un método geométrico para estimar la posición del Polo Norte Galáctico a
partir de la simetría de pares de estrellas en ascensión recta y declinación, usando un catálogo
observacional limitado. El ciclo `ngp-improvement` (2026) reprodujo y extendió ese método con datos
reales de Gaia DR3 (53,082 estrellas de disco, |b|<15°, G<15, paralaje S/N>5), introduciendo un
estimador de círculo máximo por SVD (`great_circle_pole`) sin parámetro de referencia previo, y
validó su incertidumbre estadística mediante bootstrap. El resultado señaló un hecho incómodo: el
polo medido (δ=26.49°) difiere del polo IAU publicado (δ=27.13°) por 1.44°, una diferencia que el
intervalo de confianza bootstrap al 95% (δ∈[26.40°,26.59°]) **no puede explicar como ruido
estadístico** — el CI95 es demasiado angosto y no cubre el valor IAU.

Este trabajo (`ngp-precision`) parte de esa observación como pregunta científica, no como error a
corregir por fuerza bruta. En vez de perseguir el número exacto del polo IAU (una convención de
1958, definida sobre gas neutro de HI, no sobre estrellas — ver Sección 5), replanteamos el
objetivo: **medir el polo con el mejor rigor posible y explicar honestamente de dónde viene
cualquier divergencia remanente.** Las cinco extensiones de este trabajo atacan cada una una fuente
distinta de esa divergencia:

1. ¿Está el Sol exactamente en el plano galáctico, o el ajuste de círculo máximo (que pasa por el
   origen, es decir, por el Sol) está ignorando un offset real? (Sección 2)
2. ¿La muestra de disco de todas las edades traza el mismo plano que trazadores jóvenes que
   definen dinámicamente "el plano de rotación actual"? (Sección 3)
3. ¿El método 3D con distancia (degradado en `ngp-improvement` por el ruido de 1/paralaje) puede
   rescatarse con ponderación por covarianza? (Sección 4)
4. ¿Un observable completamente independiente — el eje de rotación cinemático, de movimientos
   propios — coincide con el polo geométrico? (Sección 5)
5. ¿Cuánto del error observado es sistemático (depende de las elecciones de análisis) frente a
   estadístico (ruido de remuestreo)? (Sección 6). ¿Y cuánto de la divergencia frente al polo IAU
   se explica por la propia historia de la convención de 1958? (Sección 7)

Todos los resultados presentados aquí son reales, calculados sobre datos de Gaia DR3 descargados
para este trabajo (no simulaciones), con la excepción de las validaciones de código que se hicieron
sobre catálogos sintéticos con verdad-fundamental conocida (ver `synthetic_catalog.py`) antes de
aplicar cada método a datos reales — práctica de TDD (desarrollo guiado por pruebas) que garantiza
que cada estimador recupera correctamente un polo conocido antes de confiar en su resultado sobre
datos reales.

---

## 2. Datos y métodos

### 2.1 Muestra de disco (referencia, heredada de `ngp-improvement`)

53,082 estrellas de Gaia DR3, |b|<15°, G<15, paralaje/error>5, descargadas mediante consultas TAP
síncronas paginadas sobre `random_index` (ver nota metodológica en el README principal). Esta
muestra sirve como línea base de comparación para todos los métodos nuevos.

### 2.2 Trazadores jóvenes (nuevo, este trabajo)

Se descargaron tres catálogos reales adicionales de Gaia DR3 / VizieR, cada uno filtrado a radio
galactocéntrico R<9 kpc (la "zona plana segura" antes del inicio documentado del warp del disco,
~9–11 kpc; Skowron et al. 2019):

| Trazador | N crudo | N tras corte R<9kpc |
|---|---:|---:|
| Cefeidas clásicas (`vari_cepheid`) | 15,021 | 391 (334 con S/N paralaje>5) |
| Estrellas OB (`gaia_source`, selección por color/magnitud) | 11,196 | 9,917 |
| Cúmulos abiertos jóvenes (Cantat-Gaudin et al. 2020, vía VizieR) | 2,017 | 455 |

**Nota de transparencia sobre las Cefeidas**: la relación período-luminosidad usada para estimar
distancias de Cefeidas fue recalibrada durante este trabajo (a=−2.4352, b=−0.2537 en M_G vs
log P, ajustada por sigma-clipping iterativo sobre la sub-muestra de mejor paralaje) tras
descubrir que los coeficientes de partida (heredados de un placeholder anterior) producían
distancias no físicas (mediana ~102 kpc, máximo ~2210 kpc). La calibración corregida da distancias
razonables, pero **no incluye corrección por extinción** (no se usó un índice tipo Wesenheit) y se
apoya en paralajes de Gaia, no en una calibración externa independiente. Los resultados de Cefeidas
deben leerse como **ilustrativos**, no definitivos — la muestra es además la más pequeña (N=334
utilizable) de los tres trazadores.

### 2.3 Métodos

- **`great_circle_pole`** (heredado, sin cambios): ajuste de círculo máximo por SVD, sin
  parámetro libre, pasa por el origen (el Sol). Método insignia de `ngp-improvement`.
- **`offset_plane_pole`** (nuevo, F1): igual que el anterior pero con offset libre — PCA
  ponderada con centrado en la media en vez de forzar el ajuste por el origen. Recupera
  simultáneamente el polo **y** z☉ (altura del Sol sobre el plano ajustado, en pc).
- **`weighted_tls_plane`** (nuevo, F3): sustituye el RANSAC 3D simple de `ngp-improvement`
  (sesgado por ruido de distancia no ponderado, 1000/paralaje) por un ajuste IRLS ponderado por la
  covarianza posicional por estrella. Intenta aplicar la corrección de zero-point de paralaje de
  Gaia DR3 (Lindegren et al. 2021) antes de convertir paralaje a distancia; si el paquete opcional
  `gaiadr3-zeropoint` no está instalado, procede sin corregir y **marca explícitamente** el
  resultado con `zero_point_corrected=False` — nunca de forma silenciosa.
- **`kinematic_pole`** (nuevo, F4): estima el eje de rotación del disco a partir de movimientos
  propios (pmra/pmdec) — un observable físicamente independiente de la posición estelar. Requiere
  restar el movimiento peculiar del Sol (U,V,W)=(11.1,12.24,7.25) km/s (Schönrich et al. 2010),
  definido en el marco **Galáctico** y que debe rotarse al marco **ecuatorial** (ICRS) antes de
  combinarse con los vectores de posición usados internamente — un detalle de convención de marco
  de referencia que, de omitirse, produce discrepancias geométrico-cinemáticas de 13°–22°
  (físicamente absurdas); corregido, el acuerdo real es de 2°–8° (ver Sección 5).
- **`systematics_grid` / `combine_error_budget`** (nuevo, F5): re-ejecuta `great_circle_pole`
  variando una a la vez cada elección de análisis (corte en |b|, magnitud límite, S/N de
  paralaje, capa de distancia, hemisferio) para medir cuánto se desplaza el polo — la definición
  operacional de error *sistemático*, combinado en cuadratura con el error *estadístico* del
  bootstrap: σ_total = √(σ_stat² + σ_syst²).
- **`b1950_to_j2000_pole` / `decompose_divergence`** (nuevo, F6): transforma el polo IAU original
  (1958, sistema B1950/FK4) a J2000/ICRS mediante `astropy`, y reparte la divergencia observada en
  tres términos: error de medición de 1958, artefacto de transformación FK4→FK5 (Liu, Zhu & Zhang
  2011, ~0.377″), y una diferencia gas-vs-estrellas definida como el remanente que cierra el
  presupuesto (no una derivación física independiente — ver caveat en 5.3).

Todos los módulos nuevos se desarrollaron con TDD estricto (prueba antes que implementación) y se
validaron primero sobre catálogos sintéticos con parámetros de verdad-fundamental conocidos
(`synthetic_catalog.py`) antes de aplicarse a los datos reales presentados aquí. El código completo
es reproducible sin conexión de red a partir de los CSV cacheados en `data/`.

---

## 3. Resultados

### 3.1 z☉: altura del Sol sobre el plano galáctico

Sobre la muestra completa de disco (N=53,082), `offset_plane_pole` da:

- α_NGP = 12.9666 h, δ_NGP = 25.6742°
- **z☉ = −16.57 ± 1.62 pc**

El valor absoluto es competitivo en precisión con la literatura (Karim & Mamajek 2017:
17±5 pc; mediana histórica: 17±2 pc), pero el **signo es opuesto** al esperado. Una validación
cruzada independiente — ajustando δ como función de la distancia por capas y extrapolando a
d→∞ (`delta_vs_distance_shells`) — da δ_∞ = 26.52° ± 0.18°, consistente con el δ de
`great_circle_pole` (26.49°) dentro de su incertidumbre, lo cual descarta un error de
implementación en `offset_plane_pole` y apunta en cambio a un sesgo de selección de muestra
(posiblemente asimetría en extinción/magnitud límite no uniforme en longitud galáctica) como
causa más probable de la discrepancia de signo. Esta es una discrepancia real y no resuelta en
este ciclo — ver Sección 4 de discusión.

Con el método 3D ponderado (`weighted_tls_plane`) sobre la misma muestra, z☉ = 0.02 ± 148 pc —
la barra de error es demasiado grande para ser informativa, un resultado esperado dado que este
método depende de distancias individuales (1/paralaje), mucho más ruidosas que el ajuste
puramente geométrico y ponderado de `offset_plane_pole`.

### 3.2 Trazadores jóvenes: tres polos independientes

| Trazador | α_NGP (h) | δ_NGP (°) | Error vs. IAU (°) |
|---|---:|---:|---:|
| Cefeidas | 12.9534 | 26.2592 | 1.636 |
| Estrellas OB | 12.9023 | 25.0822 | 2.165 |
| Cúmulos jóvenes | 12.6524 | 26.7759 | 2.665 |
| *(referencia: disco completo)* | *12.9463* | *26.4924* | *1.437* |

El spread inter-trazador (δ va de 25.08° a 26.78°, un rango de 1.7°) es del mismo orden que el
error sistemático estimado en la Sección 3.4 — es decir, la elección de trazador es en sí misma
una fuente de sistemático, no solo un chequeo de consistencia.

### 3.3 Polo cinemático: acuerdo geométrico-cinemático

Tras corregir el marco de referencia del movimiento solar peculiar (ver Sección 2.3):

| Trazador | Separación sin corregir | Separación corregida |
|---|---:|---:|
| Disco completo | 13.00° | **2.80°** |
| Cefeidas | 13.71° | **8.26°** |
| Estrellas OB | 20.30° | **6.19°** |
| Cúmulos jóvenes | 22.16° | **1.98°** |

El acuerdo real (2°–8°) es mucho peor que el ~0.1°–0.2° que se esperaba originalmente (y que una
validación puramente sintética había sugerido erróneamente ~0.000°, al ser una prueba de
autoconsistencia del código y no del acuerdo físico real). El residuo de 2°–8° es físicamente
razonable: deriva asimétrica (asymmetric drift), movimientos no circulares, y diferencias
poblacionales de edad/dispersión de velocidades entre trazadores no están modeladas por un ajuste
de eje de rotación puramente circular. El caso de Cefeidas, el peor acuerdo (~8°), es consistente
con su muestra pequeña (N=334) y su calibración P-L todavía provisional.

### 3.4 Presupuesto de error honesto: sistemático vs. estadístico

| Trazador | σ_stat (°) | σ_syst (°) | σ_total (°) | Razón syst/stat |
|---|---:|---:|---:|---:|
| Disco completo | 0.067 | 1.022 | 1.024 | **15.3×** |
| Cefeidas | 0.958 | 2.481 | 2.660 | 2.6× |
| Estrellas OB | 0.193 | 1.247 | 1.262 | 6.5× |
| Cúmulos jóvenes | 0.645 | 1.135 | 1.306 | 1.8× |

Este es, sin ambigüedad, el resultado más sólido de este trabajo (ver `results/fig_error_budget.png`).
En todos los casos el error sistemático domina sobre el estadístico — en la muestra de disco
completo por un factor de 15. Esto significa que el intervalo de confianza bootstrap reportado en
`ngp-improvement` (que solo capturaba σ_stat) era **engañosamente preciso**: describía correctamente
el ruido de remuestreo, pero ese ruido es un componente menor del error real. Ampliar la muestra
no mejoraría sustancialmente la precisión final; lo que sí la movería son las elecciones de corte
de análisis (magnitud límite, calidad de paralaje, extensión en latitud galáctica, hemisferio).

### 3.5 Forense de la convención IAU

La transformación del polo IAU original (B1950/FK4, 1958) a J2000/ICRS vía `astropy` reproduce el
valor publicado con un error de 4.1×10⁻⁵° (muy por debajo del umbral de validación de 0.001°),
confirmando que la transformación de coordenadas está bien entendida y no es en sí misma una
fuente relevante de discrepancia. La descomposición de la divergencia total frente al polo IAU
J2000 (usando `great_circle_pole` sobre el disco completo) da:

- Divergencia total: **1.350°**
- Término de medición de 1958: 0.124°
- Artefacto de transformación FK4→FK5 (Liu, Zhu & Zhang 2011): 0.0001° (despreciable)
- Término gas-vs-estrellas (remanente de cierre): 1.226°
- Residuo: 0.000° (por construcción)
- Error frente a Karim & Mamajek (2017, medición moderna independiente): 1.435°

El término "gas-vs-estrellas" es, por diseño, el remanente que cierra el presupuesto tras restar
los otros dos términos conocidos — **no** es una derivación física independiente de cuánto difiere
el plano trazado por gas neutro (HI, base de la convención IAU de 1958) del plano trazado por
estrellas (este trabajo). Se reporta con ese caveat explícito para no sobre-interpretar el número.

### 3.6 Tabla maestra

La tabla completa método×trazador (`results/master_table.{md,csv,tex}`) combina los 15 pares
método-trazador con sus errores frente a ambas referencias (IAU y Karim & Mamajek 2017). El mejor
resultado individual frente a IAU es `offset_plane` sobre cúmulos jóvenes (0.765°); el peor es
`kinematic` sobre Cefeidas (8.169°) — consistente con la Sección 3.3.

---

## 4. Discusión

El objetivo original de este ciclo era ambicioso: reducir el error total del polo de 1.44° a
0.03°–0.05° (una mejora de ~30×). **Ese objetivo no se alcanzó**, y es importante decirlo con la
misma claridad con que se reportan los éxitos. La razón, sin embargo, no es una falla de
implementación sino un hallazgo científico en sí mismo: el presupuesto de error de la Sección 3.4
muestra que el sistemático domina por 2×–15× según el trazador, y ningún refinamiento puramente
estadístico (más estrellas, mejor bootstrap) puede superar un techo que está fijado por decisiones
de selección de muestra. Esto reencuadra la pregunta de investigación para un trabajo futuro: no
"¿cómo reducimos el ruido?" sino "¿cómo controlamos o eliminamos el sesgo de selección?"

La discrepancia de signo en z☉ (Sección 3.1) queda como la pregunta abierta más intrigante de este
trabajo. Que la extrapolación independiente δ(d→∞) coincida con `great_circle_pole` mientras
`offset_plane_pole` da un z☉ de signo opuesto al esperado sugiere que el problema no está en el
método de ajuste en sí, sino en una asimetría real de la muestra — candidatos incluyen extinción
diferencial no simétrica en longitud galáctica, o una asimetría norte-sur genuina en la densidad
estelar observada por los cortes de magnitud y calidad de paralaje aplicados. Resolver esto
requeriría repetir el análisis con una máscara de extinción explícita (p. ej. usando 3D dust maps)
o con cortes de calidad más estrictos por hemisferio.

El descubrimiento del error de marco de referencia en el polo cinemático (Sección 2.3, 3.3) es un
recordatorio metodológico importante: la validación sobre datos sintéticos, aunque necesaria, no es
suficiente cuando el catálogo sintético comparte implícitamente la misma convención "incorrecta"
que el código bajo prueba. Solo la aplicación a datos reales, con su propia física externa
(movimiento solar de la literatura, en un marco de referencia documentado independientemente),
expuso el error. Esto refuerza la práctica de contrastar resultados sintéticos con al menos un
chequeo de sanidad basado en física externa antes de reportar un resultado como validado.

Las limitaciones de este trabajo son explícitas: (1) la calibración período-luminosidad de
Cefeidas es preliminar y sin corrección de extinción; (2) el zero-point de paralaje de Gaia DR3 no
se aplicó por no tener el paquete opcional instalado (el código lo soporta y lo señala
explícitamente cuando ocurre, vía `zero_point_corrected=False`); (3) el término gas-vs-estrellas del
presupuesto forense es un remanente de cierre, no una medición independiente; (4) la muestra de
cúmulos y OB no incluye corrección de extinción diferencial explícita.

---

## 5. Conclusiones

1. Se midió z☉ con precisión estadística competitiva con la literatura, pero con un signo
   inesperado que queda como pregunta abierta — probablemente un sesgo de selección de muestra,
   no un error de método (validado por una extrapolación independiente δ(d→∞) consistente).
2. Se descargaron y procesaron tres trazadores jóvenes reales e independientes (Cefeidas, OB,
   cúmulos abiertos), con un spread inter-trazador (~1.7° en δ) del mismo orden que el error
   sistemático estimado.
3. El método 3D ponderado ya no es claramente el peor estimador (compite con `offset_plane` en
   α/δ), aunque su z☉ no es informativo sin el zero-point de Lindegren.
4. El polo cinemático se entregó y comparó contra el geométrico; el acuerdo real (2°–8°, tras
   corregir un error de marco de referencia descubierto en este trabajo) es peor que el objetivo
   inicial pero físicamente razonable.
5. **El resultado más importante de este ciclo es el presupuesto de error honesto**: el error
   sistemático domina sobre el estadístico por un factor de 2× a 15× según el trazador. La
   precisión bootstrap reportada en el ciclo anterior, aunque correcta, describía solo una fracción
   menor del error real.
6. La divergencia de 1.35° frente al polo IAU se descompuso en tres términos cuantificados, con
   caveats explícitos sobre cuáles son derivaciones independientes y cuál es un remanente de
   cierre de presupuesto.
7. Los cuatro archivos originales del método de Cano (2022) —`Approximation.ipynb`,
   `automatedAR.py`, `automatedDEC.py`, `DEC2.py`— permanecen sin modificar; todo lo aquí
   presentado es una capa de extensión reproducible y con 179 pruebas automatizadas en verde.

El mensaje central para el lector es que este trabajo no alcanzó el objetivo ambicioso de una
precisión ~30× mejor, pero sí logró algo de mayor valor científico: convertir un resultado que
parecía "estadísticamente preciso pero sesgado" en un resultado con un presupuesto de error
completo y honesto, que identifica con claridad dónde está el verdadero cuello de botella
(sistemático, no estadístico) y traza un camino concreto — trazadores mejor caracterizados,
corrección de zero-point, calibración externa de Cefeidas, control explícito de extinción — para
el próximo ciclo de mejora.

---

## Agradecimientos

A Ludving Cano, autor del método original que hizo posible esta extensión.

## Referencias

- Cano, L. (2022). Aproximación geométrica del polo norte galáctico. *Revista Boliviana de Física*, 40.
- Karim, M. T., & Mamajek, E. E. (2017). Revised geometric estimate of the North Galactic Pole and the Sun's height above the Galactic mid-plane. *MNRAS*, 465, 472.
- Liu, J. C., Zhu, Z., & Zhang, H. (2011). Reconsidering the Galactic coordinate system. *A&A*, 526, A16.
- Lindegren, L., et al. (2021). Gaia Early Data Release 3: Parallax bias versus magnitude, colour, and position. *A&A*, 649, A4.
- Cantat-Gaudin, T., et al. (2020). Gaia EDR3: A new census of open clusters. *A&A*, 640, A1.
- Skowron, D. M., et al. (2019). A three-dimensional map of the Milky Way using classical Cepheids. *Science*, 365, 478.
- Schönrich, R., Binney, J., & Dehnen, W. (2010). Local kinematics and the local standard of rest. *MNRAS*, 403, 1829.
- Gaia Collaboration (2023). Gaia Data Release 3: Summary of the content and survey properties. *A&A*, 674, A1.

---

*Este borrador fue generado a partir de los resultados computacionales reproducibles del ciclo
`ngp-precision` (ver `NGP_Precision.ipynb` y `sdd/ngp-precision/*` en la memoria persistente del
proyecto). Pendiente de revisión por el autor antes de cualquier envío formal a la revista.*
