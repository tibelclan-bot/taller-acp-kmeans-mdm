"""Carga y selección de los datos de la MDM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mdm import config


def load_panel(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Lee el panel municipio-año conservando los códigos DANE como texto."""
    return pd.read_csv(path, dtype=dict.fromkeys(config.CODE_COLUMNS, "string"))


def available_years(panel: pd.DataFrame) -> list[int]:
    """Años presentes en el panel, del más antiguo al más reciente."""
    return sorted(panel[config.YEAR].unique().tolist())


def select_year(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """Devuelve el corte transversal de un año, indexado por código de municipio."""
    cross_section = panel.loc[panel[config.YEAR] == year]
    return cross_section.set_index(config.MUNICIPALITY_CODE)
