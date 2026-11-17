"""Pantalla 2: estructura de la base, estadísticas descriptivas y correlaciones."""

from __future__ import annotations

import streamlit as st

from mdm import charts, config
from views.context import ViewContext


def render(context: ViewContext) -> None:
    analysis = context.analysis
    raw, variables = analysis.raw, analysis.variables

    st.title("Exploración de los datos")
    st.caption(f"Corte transversal de {analysis.settings.year}, antes de cualquier limpieza.")

    left, middle, right = st.columns(3)
    left.metric("Registros (municipios)", len(raw))
    middle.metric("Columnas en la base", raw.shape[1] + 1)
    right.metric("Variables en el modelo", len(variables))

    st.subheader("Vista de la base")
    st.dataframe(raw.reset_index(), hide_index=True, width="stretch", height=260)

    with st.expander("Tipos de dato por columna"):
        types = raw.reset_index().dtypes.astype(str).rename("Tipo").rename_axis("Columna")
        st.dataframe(types.reset_index(), hide_index=True, width="stretch")

    st.subheader("Estadísticas descriptivas")
    description = raw[variables].describe().T.rename(index=charts.label)
    st.dataframe(description.style.format("{:.2f}"), width="stretch")
    st.plotly_chart(charts.distribution_boxes(raw, variables), width="stretch")
    st.markdown(
        """
        **Cómo leerlo.** Aunque todas las dimensiones comparten la escala 0–100, su dispersión es
        muy distinta: Salud y Seguridad se concentran en puntajes altos con poca variación,
        mientras que Movilización de recursos es baja en la mayoría de municipios y tiene una cola
        larga de valores altos. Sin estandarizar, las dimensiones más dispersas dominarían tanto
        el ACP como las distancias de K-Means.
        """
    )

    st.subheader("Relaciones entre dimensiones")
    st.plotly_chart(charts.correlation_heatmap(raw, variables), width="stretch")
    st.markdown(
        """
        **Cómo leerlo.** Azul indica que dos dimensiones suben juntas; rojo, que una sube cuando
        la otra baja. Las correlaciones moderadas entre varias dimensiones justifican el ACP: hay
        información compartida que puede resumirse. Las que casi no se correlacionan con el resto
        aportan información propia y probablemente necesiten su propio componente.
        """
    )

    st.subheader(f"Municipio seleccionado: {context.highlight[config.MUNICIPALITY]}")
    comparison = (
        raw[variables]
        .mean()
        .rename("Promedio nacional")
        .to_frame()
        .assign(**{context.highlight[config.MUNICIPALITY]: context.highlight[variables]})
        .rename(index=charts.label)
    )
    st.dataframe(comparison.style.format("{:.1f}"), width="stretch")
