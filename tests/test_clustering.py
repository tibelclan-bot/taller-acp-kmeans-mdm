from __future__ import annotations

from string import ascii_uppercase

import pandas as pd
import pytest

from mdm import clustering, config
from mdm.pipeline import Analysis, Settings, run_analysis

MANUAL_K = 4


def test_every_municipality_gets_a_segment(analysis: Analysis) -> None:
    labels = analysis.clustering.labels

    assert labels.index.equals(analysis.clean.index)
    assert labels.nunique() == analysis.k


def test_segments_are_lettered_from_largest_to_smallest(analysis: Analysis) -> None:
    sizes = analysis.clustering.labels.value_counts()

    assert sizes.index.tolist() == list(ascii_uppercase[: analysis.k])


def test_inertia_decreases_as_k_grows(analysis: Analysis) -> None:
    assert analysis.k_evaluation[clustering.INERTIA].is_monotonic_decreasing


def test_suggested_k_has_the_highest_silhouette(analysis: Analysis) -> None:
    silhouettes = analysis.k_evaluation[clustering.SILHOUETTE]

    assert silhouettes[analysis.suggested_k] == silhouettes.max()


def test_manual_k_overrides_the_suggestion(panel: pd.DataFrame, default_settings: Settings) -> None:
    settings = Settings(
        year=default_settings.year, variables=default_settings.variables, manual_k=MANUAL_K
    )

    assert run_analysis(panel, settings).k == MANUAL_K


def test_results_are_reproducible(panel: pd.DataFrame, default_settings: Settings) -> None:
    first = run_analysis(panel, default_settings).clustering.labels
    second = run_analysis(panel, default_settings).clustering.labels

    pd.testing.assert_series_equal(first, second)


def test_profile_sizes_add_up_to_all_municipalities(analysis: Analysis) -> None:
    assert analysis.segment_profile[clustering.SIZE].sum() == len(analysis.clean)


def test_crosstab_rows_are_percentages(analysis: Analysis) -> None:
    crosstab = clustering.cross_tabulate(
        analysis.clustering.labels, analysis.clean[config.PEER_GROUP]
    )

    assert crosstab.sum(axis=1).tolist() == pytest.approx([100] * len(crosstab))
