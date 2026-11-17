from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mdm import config, data, preprocessing
from tests.conftest import INCOMPLETE_YEAR

VARIABLES = ["a", "b"]


@pytest.fixture
def dirty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [10.0, 10.0, np.nan, 100.4, 50.0], "b": [20.0, 20.0, 30.0, 40.0, 60.0]},
        index=["m1", "m1", "m2", "m3", "m4"],
    )


def test_quality_report_counts_each_problem(dirty_frame: pd.DataFrame) -> None:
    report = preprocessing.assess_quality(dirty_frame, VARIABLES)

    assert report.duplicated_rows == 1
    assert report.rows_with_missing == 1
    assert report.out_of_range_by_variable["a"] == 1


def test_clean_removes_duplicates_and_missing_and_clips(dirty_frame: pd.DataFrame) -> None:
    cleaned = preprocessing.clean(dirty_frame, VARIABLES)

    assert cleaned.index.tolist() == ["m1", "m3", "m4"]
    assert cleaned["a"].max() == config.SCORE_MAXIMUM


def test_clean_does_not_modify_its_input(dirty_frame: pd.DataFrame) -> None:
    original = dirty_frame.copy()

    preprocessing.clean(dirty_frame, VARIABLES)

    pd.testing.assert_frame_equal(dirty_frame, original)


def test_standardize_gives_zero_mean_and_unit_deviation(dirty_frame: pd.DataFrame) -> None:
    cleaned = preprocessing.clean(dirty_frame, VARIABLES)

    standardized = preprocessing.standardize(cleaned, VARIABLES)

    assert np.allclose(standardized.mean(), 0)
    assert np.allclose(standardized.std(ddof=0), 1)


def test_real_incomplete_year_loses_only_the_unmeasured_municipality(panel: pd.DataFrame) -> None:
    variables = list(config.DIMENSIONS)
    cross_section = data.select_year(panel, INCOMPLETE_YEAR)

    cleaned = preprocessing.clean(cross_section, variables)

    assert len(cross_section) - len(cleaned) == 1
