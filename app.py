"""Punto de entrada de la aplicación Streamlit.

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from views import (
    components,
    conclusions,
    exploration,
    introduction,
    preparation,
    segmentation,
)
from views.context import ViewContext, build_context

# (función que dibuja la pantalla, título, ícono, ruta en la URL)
PAGES: tuple[tuple[Callable[[ViewContext], None], str, str, str], ...] = (
    (introduction.render, "Contexto", ":material/menu_book:", "contexto"),
    (exploration.render, "Exploración", ":material/search:", "exploracion"),
    (preparation.render, "Preparación", ":material/cleaning_services:", "preparacion"),
    (components.render, "ACP", ":material/scatter_plot:", "acp"),
    (segmentation.render, "K-Means", ":material/hub:", "kmeans"),
    (conclusions.render, "Conclusiones", ":material/lightbulb:", "conclusiones"),
)


def main() -> None:
    st.set_page_config(page_title="Perfiles de desempeño municipal", page_icon="🏛️", layout="wide")
    context = build_context()
    pages = [
        st.Page(lambda render=render: render(context), title=title, icon=icon, url_path=path)
        for render, title, icon, path in PAGES
    ]
    st.navigation(pages).run()


main()
