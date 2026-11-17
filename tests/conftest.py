"""Datos compartidos por las pruebas."""

from __future__ import annotations

import pandas as pd
import pytest

from mdm import config, data
from mdm.pipeline import Analysis, Settings, run_analysis

LATEST_YEAR = 2024
INCOMPLETE_YEAR = 2016  # Barrancominas aún no tenía medición


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return data.load_panel()


@pytest.fixture(scope="session")
def default_settings() -> Settings:
    return Settings(year=LATEST_YEAR, variables=tuple(config.DIMENSIONS))


@pytest.fixture(scope="session")
def analysis(panel: pd.DataFrame, default_settings: Settings) -> Analysis:
    return run_analysis(panel, default_settings)
