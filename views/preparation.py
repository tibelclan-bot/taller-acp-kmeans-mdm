"""Pantalla 3: diagnóstico de calidad, decisiones de limpieza y estandarización."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mdm import charts, config
from views.context import ViewContext

DECISIONS = (
    (
        "Identificadores (códigos, nombres)",
        "No entran al modelo",
        "No miden desempeño; solo identifican.",
    ),
    (
        "Índice MDM, Gestión y Resultados",
        "No entran al modelo",
        "Son combinaciones de las dimensiones: duplicarían información.",
    ),
    ("Registros duplicados", "Se eliminan", "Un municipio debe pesar una sola vez."),
    (
        "Registros con faltantes",
        "Se eliminan",
        "Son casos aislados; imputarlos inventaría un desempeño que no se midió.",
    ),
    (
        "Puntajes fuera de 0–100",
        "Se acotan al rango",
        "La escala teórica es 0–100 y los excesos observados son de décimas.",
    ),
    (
        "Escalas con distinta dispersión",
        "Estandarización (media 0, desviación 1)",
        "ACP y K-Means dependen de varianzas y distancias.",
    ),
)


def _quality_table(context: ViewContext) -> pd.DataFrame:
    quality = context.analysis.quality
    table = pd.DataFrame(
        {
            "Valores faltantes": quality.missing_by_variable,
            "Fuera del rango 0–100": quality.out_of_range_by_variable,
        }
    )
    return table.rename(index=charts.label)


def _scale_check(context: ViewContext) -> pd.DataFrame:
    analysis = context.analysis
    table = pd.DataFrame(
        {
            "Media original": analysis.clean[analysis.variables].mean(),
            "Desv. estándar original": analysis.clean[analysis.variables].std(ddof=0),
            "Media estandarizada": analysis.standardized.mean(),
            "Desv. estándar estandarizada": analysis.standardized.std(ddof=0),
        }
    )
    return table.rename(index=charts.label)


def render(context: ViewContext) -> None:
    analysis = context.analysis
    quality = analysis.quality
    removed = quality.rows - len(analysis.clean)

    st.title("Preparación de los datos")

    st.subheader("1. Diagnóstico de calidad")
    first, second, third = st.columns(3)
    first.metric("Registros duplicados", quality.duplicated_rows)
    second.metric("Registros con faltantes", quality.rows_with_missing)
    third.metric("Puntajes fuera de rango", int(quality.out_of_range_by_variable.sum()))
    st.dataframe(_quality_table(context), width="stretch")

    st.subheader("2. Decisiones")
    st.table(pd.DataFrame(DECISIONS, columns=["Problema", "Decisión", "Razón"]))
    st.success(
        f"**{len(analysis.clean)} municipios** entran al análisis "
        f"({removed} eliminados de {quality.rows})."
    )

    st.subheader("3. Estandarización")
    st.latex(r"z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}")
    st.markdown(
        r"""
        Donde $x_{ij}$ es el puntaje del municipio $i$ en la dimensión $j$, $\bar{x}_j$ es el
        promedio nacional de esa dimensión y $s_j$ su desviación estándar. Después de
        transformar, un valor de $+1$ significa «una desviación estándar por encima del promedio
        nacional» en cualquier dimensión.
        """
    )
    st.dataframe(_scale_check(context).style.format("{:.2f}"), width="stretch")

    st.subheader(f"Ejemplo: {context.highlight[config.MUNICIPALITY]}")
    example = pd.DataFrame(
        {
            "Puntaje original": context.highlight[analysis.variables],
            "Puntaje estandarizado (z)": analysis.standardized.loc[context.highlight.name],
        }
    ).rename(index=charts.label)
    st.dataframe(example.style.format("{:.2f}"), width="stretch")
