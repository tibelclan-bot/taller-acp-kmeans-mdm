"""Constantes del proyecto: rutas, variables y parámetros de los modelos."""

from __future__ import annotations

from pathlib import Path

AUTHOR = "Juan Diego Zutta"
COURSE = "Machine Learning aplicado a Economía y Finanzas"
UNIVERSITY = "Pontificia Universidad Javeriana Cali"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mdm_municipios_2016_2024.csv"

MUNICIPALITY_CODE = "codigo_municipio"
MUNICIPALITY = "municipio"
DEPARTMENT_CODE = "codigo_departamento"
DEPARTMENT = "departamento"
LATITUDE = "latitud"
LONGITUDE = "longitud"
YEAR = "anio"
PEER_GROUP = "grupo_dotaciones"
OVERALL_INDEX = "mdm"
CLUSTER = "segmento"

CODE_COLUMNS = (MUNICIPALITY_CODE, DEPARTMENT_CODE)

MANAGEMENT = "Gestión"
RESULTS = "Resultados"

# Las ocho dimensiones de la MDM: columna -> (etiqueta legible, componente del DNP).
DIMENSIONS: dict[str, tuple[str, str]] = {
    "movilizacion_recursos": ("Movilización de recursos", MANAGEMENT),
    "ejecucion_recursos": ("Ejecución de recursos", MANAGEMENT),
    "gobierno_abierto": ("Gobierno abierto", MANAGEMENT),
    "ordenamiento_territorial": ("Ordenamiento territorial", MANAGEMENT),
    "educacion": ("Educación", RESULTS),
    "salud": ("Salud", RESULTS),
    "servicios_publicos": ("Servicios públicos", RESULTS),
    "seguridad_convivencia": ("Seguridad y convivencia", RESULTS),
}
DIMENSION_LABELS: dict[str, str] = {column: label for column, (label, _) in DIMENSIONS.items()}

SCORE_MINIMUM = 0.0
SCORE_MAXIMUM = 100.0

MINIMUM_VARIABLES = 4
KAISER_EIGENVALUE = 1.0
DEFAULT_VARIANCE_THRESHOLD = 0.70

RANDOM_STATE = 42
KMEANS_INITIALIZATIONS = 10
MINIMUM_K = 2
MAXIMUM_K = 10

DEFAULT_HIGHLIGHT = "Pasto"
