"""Resúmenes que traducen los resultados numéricos en hallazgos legibles."""

from __future__ import annotations

import pandas as pd


def dominant_variables(loadings: pd.DataFrame, component: str, count: int) -> pd.Series:
    """Variables con mayor carga absoluta en un componente, con su signo."""
    order = loadings[component].abs().sort_values(ascending=False).index
    return loadings.loc[order[:count], component]


def distinctive_variables(
    standardized_profile: pd.DataFrame, segment: str, count: int
) -> pd.Series:
    """Variables en las que un segmento se aleja más del promedio nacional."""
    deviations = standardized_profile.loc[segment]
    order = deviations.abs().sort_values(ascending=False).index
    return deviations.loc[order[:count]]


def share_below(values: pd.Series, threshold: float) -> float:
    """Proporción de observaciones por debajo de un umbral."""
    return float((values < threshold).mean())
