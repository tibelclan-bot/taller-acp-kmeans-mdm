from __future__ import annotations

import pandas as pd

from mdm import config, data


def test_panel_keeps_dane_codes_with_leading_zeros(panel: pd.DataFrame) -> None:
    assert panel[config.MUNICIPALITY_CODE].str.len().eq(5).all()


def test_panel_has_one_row_per_municipality_and_year(panel: pd.DataFrame) -> None:
    assert not panel.duplicated([config.MUNICIPALITY_CODE, config.YEAR]).any()


def test_select_year_returns_only_that_year(panel: pd.DataFrame) -> None:
    year = data.available_years(panel)[-1]

    cross_section = data.select_year(panel, year)

    assert cross_section[config.YEAR].eq(year).all()
    assert cross_section.index.is_unique
