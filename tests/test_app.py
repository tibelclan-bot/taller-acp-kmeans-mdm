"""Pruebas de humo: cada pantalla se dibuja sin errores con la configuración inicial."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

VIEWS = ["introduction", "exploration", "preparation", "components", "segmentation", "conclusions"]
SCRIPT = """
from views import {view}
from views.context import build_context

{view}.render(build_context())
"""
TIMEOUT_SECONDS = 120


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders_without_errors(view: str) -> None:
    app = AppTest.from_string(SCRIPT.format(view=view), default_timeout=TIMEOUT_SECONDS).run()

    assert not app.exception
    assert not app.error


def test_entry_point_runs() -> None:
    app = AppTest.from_file("../app.py", default_timeout=TIMEOUT_SECONDS).run()

    assert not app.exception
