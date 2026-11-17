"""Segmentación con K-Means: comparación de K, ajuste y perfiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from string import ascii_uppercase

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from mdm import config

K = "k"
INERTIA = "inercia"
SILHOUETTE = "silueta"
SIZE = "municipios"


@dataclass(frozen=True)
class ClusteringResult:
    """Asignación de segmentos y su calidad."""

    labels: pd.Series  # letra del segmento por observación
    sample_silhouettes: pd.Series
    silhouette: float
    inertia: float

    @property
    def segments(self) -> list[str]:
        return sorted(self.labels.unique())


def _build_model(k: int) -> KMeans:
    return KMeans(
        n_clusters=k,
        n_init=config.KMEANS_INITIALIZATIONS,
        random_state=config.RANDOM_STATE,
    )


def evaluate_k(features: pd.DataFrame, k_values: Iterable[int]) -> pd.DataFrame:
    """Calcula inercia (codo) y silueta media para cada K candidato."""
    rows = []
    for k in k_values:
        model = _build_model(k).fit(features)
        rows.append(
            {
                K: k,
                INERTIA: model.inertia_,
                SILHOUETTE: silhouette_score(features, model.labels_),
            }
        )
    return pd.DataFrame(rows).set_index(K)


def best_k_by_silhouette(evaluation: pd.DataFrame) -> int:
    """K con la mayor silueta media."""
    return int(evaluation[SILHOUETTE].idxmax())


def _letter_labels(numeric_labels: pd.Series) -> pd.Series:
    """Renombra los clústeres con letras, de mayor a menor tamaño.

    K-Means numera los grupos de forma arbitraria. Las letras evitan leer ese
    número como un orden y hacen estable el nombre de cada segmento.
    """
    by_size = numeric_labels.value_counts().index
    letters = {cluster: ascii_uppercase[position] for position, cluster in enumerate(by_size)}
    return numeric_labels.map(letters).rename(config.CLUSTER)


def fit_kmeans(features: pd.DataFrame, k: int) -> ClusteringResult:
    """Ajusta K-Means y devuelve etiquetas y métricas de cohesión y separación."""
    model = _build_model(k).fit(features)
    numeric_labels = pd.Series(model.labels_, index=features.index)
    return ClusteringResult(
        labels=_letter_labels(numeric_labels),
        sample_silhouettes=pd.Series(
            silhouette_samples(features, model.labels_), index=features.index, name=SILHOUETTE
        ),
        silhouette=float(silhouette_score(features, model.labels_)),
        inertia=float(model.inertia_),
    )


def profile_segments(frame: pd.DataFrame, labels: pd.Series, variables: list[str]) -> pd.DataFrame:
    """Promedio de cada variable por segmento, más el tamaño del segmento."""
    grouped = frame[variables].groupby(labels)
    return grouped.mean().assign(**{SIZE: grouped.size()})


def cross_tabulate(labels: pd.Series, reference: pd.Series) -> pd.DataFrame:
    """Porcentaje de cada categoría de referencia que cae en cada segmento."""
    return pd.crosstab(reference, labels, normalize="index") * 100
