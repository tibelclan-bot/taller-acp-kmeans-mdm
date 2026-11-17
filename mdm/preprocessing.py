"""Diagnóstico de calidad, limpieza y estandarización de las variables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mdm import config


@dataclass(frozen=True)
class QualityReport:
    """Problemas de calidad encontrados antes de limpiar."""

    rows: int
    duplicated_rows: int
    missing_by_variable: pd.Series
    out_of_range_by_variable: pd.Series

    @property
    def rows_with_missing(self) -> int:
        return int(self.missing_by_variable.max())


def assess_quality(frame: pd.DataFrame, variables: list[str]) -> QualityReport:
    """Cuenta duplicados, faltantes y puntajes fuera del rango teórico 0-100."""
    values = frame[variables]
    out_of_range = (values < config.SCORE_MINIMUM) | (values > config.SCORE_MAXIMUM)
    return QualityReport(
        rows=len(frame),
        duplicated_rows=int(frame.index.duplicated().sum()),
        missing_by_variable=values.isna().sum(),
        out_of_range_by_variable=out_of_range.sum(),
    )


def clean(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Elimina duplicados y filas incompletas, y acota los puntajes a 0-100."""
    deduplicated = frame.loc[~frame.index.duplicated()]
    complete = deduplicated.dropna(subset=variables).copy()
    complete[variables] = complete[variables].clip(config.SCORE_MINIMUM, config.SCORE_MAXIMUM)
    return complete


def standardize(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Lleva cada variable a media 0 y desviación estándar 1."""
    standardized = StandardScaler().fit_transform(frame[variables])
    return pd.DataFrame(standardized, index=frame.index, columns=variables)
