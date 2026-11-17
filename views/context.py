"""Estado compartido entre pantallas: datos en caché, controles y contexto."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from mdm import config, data
from mdm.charts import label
from mdm.pipeline import Analysis, ComponentRule, Settings, run_analysis


@dataclass(frozen=True)
class ViewContext:
    """Lo que cada pantalla necesita para dibujarse."""

    analysis: Analysis
    highlight: pd.Series  # fila del municipio que el usuario quiere ubicar


@st.cache_data
def get_panel() -> pd.DataFrame:
    return data.load_panel()


@st.cache_data(show_spinner="Ejecutando ACP y K-Means…")
def get_analysis(settings: Settings) -> Analysis:
    return run_analysis(get_panel(), settings)


def _select_variables() -> tuple[str, ...]:
    variables = st.multiselect(
        "Dimensiones que entran al modelo",
        options=list(config.DIMENSIONS),
        default=list(config.DIMENSIONS),
        format_func=label,
        help="Se requieren al menos cuatro variables cuantitativas.",
    )
    if len(variables) < config.MINIMUM_VARIABLES:
        st.error(f"Selecciona al menos {config.MINIMUM_VARIABLES} dimensiones para continuar.")
        st.stop()
    return tuple(variables)


def _select_component_settings(n_variables: int) -> dict:
    rule = st.radio("Criterio para conservar componentes", list(ComponentRule))
    if rule is ComponentRule.CUMULATIVE_VARIANCE:
        threshold = st.slider(
            "Varianza acumulada mínima", 0.50, 0.95, config.DEFAULT_VARIANCE_THRESHOLD, 0.05
        )
        return {"component_rule": rule, "variance_threshold": threshold}
    if rule is ComponentRule.MANUAL:
        components = st.slider("Número de componentes", 1, n_variables, 2)
        return {"component_rule": rule, "manual_components": components}
    return {"component_rule": rule}


def _select_k() -> int | None:
    if st.toggle("Usar el K con mayor silueta", value=True):
        return None
    return st.slider("Número de segmentos (K)", config.MINIMUM_K, config.MAXIMUM_K, 3)


def _select_highlight(clean: pd.DataFrame) -> pd.Series:
    names = clean[config.MUNICIPALITY] + " (" + clean[config.DEPARTMENT] + ")"
    default_matches = clean.index[clean[config.MUNICIPALITY] == config.DEFAULT_HIGHLIGHT]
    default_position = clean.index.get_loc(default_matches[0]) if len(default_matches) else 0
    code = st.selectbox(
        "Municipio a ubicar en los gráficos",
        options=clean.index,
        index=int(default_position),
        format_func=names.get,
    )
    return clean.loc[code]


def build_context() -> ViewContext:
    """Dibuja los controles laterales y ejecuta el análisis con esas decisiones."""
    panel = get_panel()
    with st.sidebar:
        st.header("Decisiones del análisis")
        years = data.available_years(panel)
        year = st.selectbox("Año de la medición", years, index=len(years) - 1)
        variables = _select_variables()
        st.subheader("ACP")
        component_settings = _select_component_settings(len(variables))
        st.subheader("K-Means")
        manual_k = _select_k()

        settings = Settings(year=year, variables=variables, manual_k=manual_k, **component_settings)
        analysis = get_analysis(settings)

        st.subheader("Exploración")
        highlight = _select_highlight(analysis.clean)
        st.caption(f"{config.AUTHOR} · {config.UNIVERSITY}")
    return ViewContext(analysis=analysis, highlight=highlight)
