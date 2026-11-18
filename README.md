# Perfiles de desempeño municipal en Colombia: ACP y K-Means

**Autor:** Juan Diego Zutta
**Curso:** Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali
**Taller individual:** aprendizaje no supervisado (ACP y K-Means) con Python, Streamlit y GitHub

- Aplicación desplegada: https://perfiles-municipales-zutta.streamlit.app


## Contexto y pregunta

El Departamento Nacional de Planeación (DNP) resume el desempeño de cada alcaldía en un índice,
la Medición del Desempeño Municipal (MDM). Un solo número sirve para ordenar municipios, pero no
muestra *cómo* se llega a él: dos municipios con el mismo puntaje pueden tener perfiles opuestos.

> **¿Qué perfiles de desempeño existen entre los municipios colombianos y qué tanto coinciden con
> los grupos de capacidades iniciales con los que el DNP los compara?**

Identificar esos perfiles es útil para diferenciar la asistencia técnica: no necesita lo mismo un
municipio que falla en transparencia que uno que falla en recaudo.

## Datos

| | |
|---|---|
| Fuente | DNP, Medición del Desempeño Municipal (MDM) |
| Cobertura | 1.102 municipios de los 32 departamentos y Bogotá, vigencias 2016 a 2024 |
| Unidad de observación | Un municipio en un año (9.918 filas) |
| Archivo | `data/mdm_municipios_2016_2024.csv` |

**Variables del modelo** (escala 0 a 100; mayor puntaje indica mejor desempeño):

| Componente del DNP | Dimensión | Columna |
|---|---|---|
| Gestión | Movilización de recursos | `movilizacion_recursos` |
| Gestión | Ejecución de recursos | `ejecucion_recursos` |
| Gestión | Gobierno abierto | `gobierno_abierto` |
| Gestión | Ordenamiento territorial | `ordenamiento_territorial` |
| Resultados | Educación | `educacion` |
| Resultados | Salud | `salud` |
| Resultados | Servicios públicos | `servicios_publicos` |
| Resultados | Seguridad y convivencia | `seguridad_convivencia` |

**Columnas que no entran al modelo:** códigos DANE y nombres (identificadores), coordenadas (solo
para el mapa), `mdm`, `gestion` y `resultados` (son combinaciones de las ocho dimensiones) y
`grupo_dotaciones` (grupo de capacidades iniciales del DNP, reservado para contrastar los
segmentos).

**Limitaciones de la fuente**

- Cada dimensión es un índice compuesto: el análisis hereda las ponderaciones del DNP.
- Los insumos son en buena parte registros administrativos reportados por las entidades.
- El CSV se construyó con `scripts/build_dataset.py` a partir de una copia pública de la base del
  DNP (panel [miltonrojasb/MDM](https://github.com/miltonrojasb/MDM)), no del archivo descargado
  directamente del portal del DNP. Como control, se contrastaron valores puntuales con
  publicaciones del DNP y coinciden (por ejemplo, Cali 86,45 y Bogotá 84,08 en 2020; promedio
  nacional de 52,8 en 2022). La vigencia 2024 no se pudo contrastar de la misma forma, así que
  conviene revisarla contra el visor oficial antes de usar las cifras para decisiones reales.
- Es información pública y agregada por municipio; no contiene datos personales.

## Por qué ACP y K-Means

- **ACP.** Las dimensiones están correlacionadas (por ejemplo, movilización de recursos con
  ordenamiento territorial, r = 0,55). El ACP muestra cuántas dimensiones independientes hay
  detrás de la tabla y qué variables las definen, sin asumir la división Gestión/Resultados.
- **K-Means.** El DNP agrupa los municipios *antes* de medirlos, por su punto de partida.
  K-Means los agrupa *por lo que obtuvieron*. Comparar ambas clasificaciones muestra qué tanto
  depende el desempeño de las capacidades iniciales.

## Metodología

1. **Exploración:** estructura, tipos, estadísticas descriptivas, diagramas de caja y matriz de
   correlaciones.
2. **Preparación:** se eliminan duplicados y registros con faltantes (solo Barrancominas, Guainía,
   sin medición de 2016 a 2020), se acotan los puntajes al rango 0–100 (en 2024 hay 14 puntajes de
   seguridad apenas por encima de 100) y se estandariza con `StandardScaler`.
3. **ACP:** se ajusta con todos los componentes; el número a conservar se decide con el criterio
   de Kaiser (autovalor > 1), con un umbral de varianza acumulada o manualmente.
4. **K-Means:** se aplica sobre los componentes conservados, se compara K de 2 a 10 con el método
   del codo y el coeficiente de silueta, y se usa el K de mayor silueta (o uno manual).
   Semilla fija (`random_state = 42`, `n_init = 10`) para que los resultados sean reproducibles.
5. **Interpretación:** los segmentos se describen con las variables originales y se contrastan
   con los grupos de capacidades iniciales del DNP, que no entraron al modelo.

Los segmentos se nombran con letras (A, B, …) del más grande al más pequeño, para no leer el
número arbitrario de K-Means como un orden.

## Resultados principales (2024, ocho dimensiones)

**ACP.** Tres componentes tienen autovalor mayor que 1 y retienen el **62,7 %** de la varianza.

| Componente | Autovalor | Varianza | Acumulada | Variables con mayor carga |
|---|---|---|---|---|
| CP1 | 2,70 | 33,8 % | 33,8 % | Movilización (+0,80), Seguridad (−0,70), Servicios públicos (+0,70), Ordenamiento (+0,69) |
| CP2 | 1,29 | 16,1 % | 49,8 % | Salud (+0,82), Educación (+0,49) |
| CP3 | 1,03 | 12,9 % | 62,7 % | Ejecución de recursos (+0,80), Gobierno abierto (+0,57) |

- **CP1, capacidad fiscal y urbana:** los municipios que recaudan más, ordenan su territorio y
  tienen mejores servicios también tienen peor puntaje en seguridad. Es el eje urbano-rural.
- **CP2, resultados sociales:** salud y educación, casi independientes del eje anterior.
- **CP3, gestión administrativa:** ejecución del presupuesto y transparencia.

La caída de la varianza es gradual: el plano CP1–CP2 resume solo la mitad de la información. El
desempeño municipal no se reduce a dos dimensiones.

**K-Means.** La silueta es máxima en K = 2 (0,297) y cae a 0,24 para K ≥ 3; el codo no muestra un
quiebre marcado.

| Segmento | Municipios | MDM promedio | Movilización | Ordenamiento | Servicios | Seguridad |
|---|---|---|---|---|---|---|
| A | 676 (61 %) | 50,1 | 13,8 | 35,8 | 40,5 | 93,7 |
| B | 426 (39 %) | 61,9 | 34,8 | 53,0 | 55,2 | 85,1 |

- **Segmento A:** municipios con baja capacidad fiscal y de ordenamiento, menor cobertura de
  servicios y mejor puntaje de seguridad.
- **Segmento B:** municipios con mayor recaudo propio, mejor ordenamiento y servicios, y peor
  seguridad. Incluye las 18 ciudades y el 87 % del grupo G1.

Frente a los grupos del DNP: el 81 % del G4 y el 82 % del G5 caen en el segmento A; los grupos
intermedios se reparten (G2: 58 % en B; G3: 65 % en A).

**Lectura.** Los segmentos no son grupos nítidos: una silueta de 0,30 indica una estructura débil,
un continuo con dos polos. La segmentación reproduce en buena medida la desigualdad de
capacidades iniciales, lo que respalda que el DNP compare municipios dentro de grupos de pares.
Los casos interesantes para política pública son los grupos intermedios (G2 y G3), donde
municipios con puntos de partida similares terminan en perfiles distintos.

Con K = 3 (silueta 0,241) aparece un matiz útil: el grupo de baja capacidad se divide entre
municipios con baja transparencia y ordenamiento pero buena salud (441) y municipios con rezago
en salud y educación (358). Es una lectura más rica, pero con fronteras más difusas.

**Limitaciones.** Segmentos de frontera difusa (la etiqueta de un municipio cercano al límite
puede cambiar con otro año o K); K-Means supone grupos compactos y es sensible a la asimetría de
movilización de recursos; el ACP solo captura relaciones lineales; es un corte transversal que
describe perfiles, no causas.

## La aplicación

Seis pantallas: **Contexto → Exploración → Preparación → ACP → K-Means → Conclusiones**.
Todo se recalcula con las decisiones del panel lateral: año, dimensiones incluidas, criterio para
conservar componentes, valor de K y municipio a ubicar en los gráficos. La página de conclusiones
se genera a partir de la corrida actual; no hay resultados fijos ni capturas.

## Cómo ejecutar

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/tibelclan-bot/taller-acp-kmeans-mdm.git
cd taller-acp-kmeans-mdm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pruebas y estilo:

```bash
pip install -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

## Estructura del repositorio

```
app.py                  Punto de entrada: navegación entre pantallas
mdm/                    Lógica analítica pura (sin Streamlit)
  config.py             Constantes: variables, rutas, parámetros
  data.py               Carga del panel y selección del año
  preprocessing.py      Diagnóstico de calidad, limpieza, estandarización
  pca.py                ACP: varianza, cargas, puntajes, criterios de retención
  clustering.py         K-Means: comparación de K, ajuste, perfiles
  insights.py           Resúmenes para interpretar resultados
  pipeline.py           Encadena todo el análisis a partir de las decisiones
  charts.py             Figuras de Plotly
views/                  Una pantalla de Streamlit por módulo
tests/                  Pruebas de la lógica y de cada pantalla
scripts/build_dataset.py  Cómo se construyó el CSV
data/                   Datos
```

Cada tabla, métrica y gráfico de la aplicación sale de una función de `mdm/`; las pantallas de
`views/` solo las muestran y explican.
