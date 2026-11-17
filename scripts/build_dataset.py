"""Construye el CSV ordenado del proyecto a partir del extracto JSON de la MDM.

Uso:
    python scripts/build_dataset.py ruta/al/extracto.json

El extracto contiene tres tablas: ``deptos`` (departamentos), ``munis``
(municipios con coordenadas) y ``rows`` (puntajes por municipio y año).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "mdm_municipios_2016_2024.csv"

SCORE_COLUMNS = {
    "y": "anio",
    "grupo": "grupo_dotaciones",
    "mdm": "mdm",
    "g": "gestion",
    "r": "resultados",
    "mov": "movilizacion_recursos",
    "eje": "ejecucion_recursos",
    "gob": "gobierno_abierto",
    "ot": "ordenamiento_territorial",
    "edu": "educacion",
    "sal": "salud",
    "ser": "servicios_publicos",
    "seg": "seguridad_convivencia",
}
MUNICIPALITY_COLUMNS = {
    "cod": "codigo_municipio",
    "nombre": "municipio",
    "dpto": "codigo_departamento",
    "lat": "latitud",
    "lon": "longitud",
}
DEPARTMENT_COLUMNS = {"cod": "codigo_departamento", "nombre": "departamento"}


def build_dataset(extract: dict) -> pd.DataFrame:
    """Une puntajes, municipios y departamentos en una sola tabla ordenada."""
    departments = pd.DataFrame(extract["deptos"])[list(DEPARTMENT_COLUMNS)].rename(
        columns=DEPARTMENT_COLUMNS
    )
    municipalities = pd.DataFrame(extract["munis"]).rename(columns=MUNICIPALITY_COLUMNS)
    scores = pd.DataFrame(extract["rows"]).rename(
        columns={"m": "codigo_municipio", **SCORE_COLUMNS}
    )

    identity = municipalities.merge(departments, on="codigo_departamento", validate="m:1")
    dataset = identity.merge(scores, on="codigo_municipio", validate="1:m")

    ordered_columns = [
        "codigo_municipio",
        "municipio",
        "codigo_departamento",
        "departamento",
        "latitud",
        "longitud",
        *SCORE_COLUMNS.values(),
    ]
    return dataset[ordered_columns].sort_values(["anio", "codigo_municipio"])


def main(extract_path: str) -> None:
    extract = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    dataset = build_dataset(extract)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(dataset):,} filas escritas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1])
