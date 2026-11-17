from __future__ import annotations

import numpy as np
import pytest

from mdm import pca
from mdm.pipeline import Analysis


def test_variance_ratios_sum_to_one(analysis: Analysis) -> None:
    assert analysis.pca.variance[pca.VARIANCE_RATIO].sum() == pytest.approx(1)


def test_eigenvalues_sum_to_number_of_variables(analysis: Analysis) -> None:
    assert analysis.pca.variance[pca.EIGENVALUE].sum() == pytest.approx(len(analysis.variables))


def test_components_are_uncorrelated(analysis: Analysis) -> None:
    correlations = analysis.pca.scores.corr().to_numpy()

    assert np.allclose(correlations, np.eye(len(correlations)), atol=1e-8)


def test_loadings_are_variable_component_correlations(analysis: Analysis) -> None:
    variable, component = analysis.variables[0], "CP1"

    correlation = analysis.standardized[variable].corr(analysis.pca.scores[component])

    assert analysis.pca.loadings.loc[variable, component] == pytest.approx(correlation)


def test_kaiser_keeps_components_with_eigenvalue_above_one(analysis: Analysis) -> None:
    retained = pca.components_by_kaiser(analysis.pca)
    eigenvalues = analysis.pca.variance[pca.EIGENVALUE]

    assert eigenvalues.iloc[retained - 1] > 1
    assert eigenvalues.iloc[retained] <= 1


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9])
def test_cumulative_rule_is_the_minimum_that_reaches_threshold(
    analysis: Analysis, threshold: float
) -> None:
    retained = pca.components_by_cumulative_variance(analysis.pca, threshold)

    assert analysis.pca.cumulative_variance(retained) >= threshold
    assert retained == 1 or analysis.pca.cumulative_variance(retained - 1) < threshold
