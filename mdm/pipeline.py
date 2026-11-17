"""Orquesta el análisis completo a partir de las decisiones del usuario."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from mdm import clustering, config, data, pca, preprocessing


class ComponentRule(StrEnum):
    """Criterios para decidir cuántos componentes conservar."""

    KAISER = "Autovalor mayor que 1 (Kaiser)"
    CUMULATIVE_VARIANCE = "Varianza acumulada mínima"
    MANUAL = "Elección manual"


@dataclass(frozen=True)
class Settings:
    """Decisiones metodológicas que el usuario controla desde la aplicación."""

    year: int
    variables: tuple[str, ...]
    component_rule: ComponentRule = ComponentRule.KAISER
    variance_threshold: float = config.DEFAULT_VARIANCE_THRESHOLD
    manual_components: int = 2
    manual_k: int | None = None  # None: usar el K con mayor silueta


@dataclass(frozen=True)
class Analysis:
    """Todos los resultados intermedios y finales de una corrida."""

    settings: Settings
    raw: pd.DataFrame
    quality: preprocessing.QualityReport
    clean: pd.DataFrame
    standardized: pd.DataFrame
    pca: pca.PcaResult
    n_components: int
    k_evaluation: pd.DataFrame
    suggested_k: int
    k: int
    clustering: clustering.ClusteringResult

    @property
    def variables(self) -> list[str]:
        return list(self.settings.variables)

    @property
    def retained_scores(self) -> pd.DataFrame:
        """Puntajes de los componentes conservados: insumo de K-Means."""
        return self.pca.scores.iloc[:, : self.n_components]

    @property
    def retained_variance(self) -> float:
        return self.pca.cumulative_variance(self.n_components)

    @property
    def segment_profile(self) -> pd.DataFrame:
        """Promedios por segmento en la escala original (0-100)."""
        return clustering.profile_segments(self.clean, self.clustering.labels, self.variables)

    @property
    def standardized_profile(self) -> pd.DataFrame:
        """Promedios por segmento en desviaciones estándar frente al promedio nacional."""
        profile = clustering.profile_segments(
            self.standardized, self.clustering.labels, self.variables
        )
        return profile[self.variables]


def resolve_components(result: pca.PcaResult, settings: Settings) -> int:
    """Aplica el criterio elegido para fijar el número de componentes."""
    if settings.component_rule is ComponentRule.KAISER:
        return pca.components_by_kaiser(result)
    if settings.component_rule is ComponentRule.CUMULATIVE_VARIANCE:
        return pca.components_by_cumulative_variance(result, settings.variance_threshold)
    return min(settings.manual_components, len(result.components))


def run_analysis(panel: pd.DataFrame, settings: Settings) -> Analysis:
    """Ejecuta preparación, ACP y K-Means en orden y devuelve todo el recorrido."""
    variables = list(settings.variables)
    raw = data.select_year(panel, settings.year)
    quality = preprocessing.assess_quality(raw, variables)
    clean = preprocessing.clean(raw, variables)
    standardized = preprocessing.standardize(clean, variables)

    pca_result = pca.fit_pca(standardized)
    n_components = resolve_components(pca_result, settings)
    features = pca_result.scores.iloc[:, :n_components]

    k_evaluation = clustering.evaluate_k(features, range(config.MINIMUM_K, config.MAXIMUM_K + 1))
    suggested_k = clustering.best_k_by_silhouette(k_evaluation)
    k = settings.manual_k or suggested_k

    return Analysis(
        settings=settings,
        raw=raw,
        quality=quality,
        clean=clean,
        standardized=standardized,
        pca=pca_result,
        n_components=n_components,
        k_evaluation=k_evaluation,
        suggested_k=suggested_k,
        k=k,
        clustering=clustering.fit_kmeans(features, k),
    )
