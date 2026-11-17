"""Análisis de Componentes Principales sobre variables estandarizadas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mdm import config

EIGENVALUE = "autovalor"
VARIANCE_RATIO = "varianza_explicada"
CUMULATIVE_RATIO = "varianza_acumulada"


@dataclass(frozen=True)
class PcaResult:
    """Resultados del ACP con todos los componentes posibles."""

    variance: pd.DataFrame  # una fila por componente
    loadings: pd.DataFrame  # variables x componentes: correlación variable-componente
    scores: pd.DataFrame  # observaciones x componentes

    @property
    def components(self) -> list[str]:
        return self.variance.index.tolist()

    def cumulative_variance(self, n_components: int) -> float:
        """Proporción de la varianza total que retienen los primeros componentes."""
        return float(self.variance[CUMULATIVE_RATIO].iloc[n_components - 1])


def fit_pca(standardized: pd.DataFrame) -> PcaResult:
    """Ajusta el ACP y organiza varianza, cargas y puntajes en tablas etiquetadas."""
    model = PCA().fit(standardized)
    names = [f"CP{number}" for number in range(1, model.n_components_ + 1)]

    ratios = model.explained_variance_ratio_
    # Con variables estandarizadas la varianza total es igual al número de variables.
    eigenvalues = ratios * standardized.shape[1]
    variance = pd.DataFrame(
        {EIGENVALUE: eigenvalues, VARIANCE_RATIO: ratios, CUMULATIVE_RATIO: ratios.cumsum()},
        index=names,
    )
    loadings = pd.DataFrame(
        model.components_.T * np.sqrt(eigenvalues), index=standardized.columns, columns=names
    )
    scores = pd.DataFrame(model.transform(standardized), index=standardized.index, columns=names)
    return PcaResult(variance=variance, loadings=loadings, scores=scores)


def components_by_kaiser(result: PcaResult) -> int:
    """Componentes cuyo autovalor supera 1: explican más que una variable original."""
    retained = int((result.variance[EIGENVALUE] > config.KAISER_EIGENVALUE).sum())
    return max(retained, 1)


def components_by_cumulative_variance(result: PcaResult, threshold: float) -> int:
    """Mínimo número de componentes cuya varianza acumulada alcanza el umbral."""
    reached = result.variance[CUMULATIVE_RATIO].to_numpy() >= threshold
    return int(reached.argmax()) + 1 if reached.any() else len(reached)
