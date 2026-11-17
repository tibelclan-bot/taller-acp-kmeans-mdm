"""Pantalla 4: Análisis de Componentes Principales."""

from __future__ import annotations

import streamlit as st

from mdm import charts, config, insights, pca
from mdm.pipeline import ComponentRule
from views.context import ViewContext

DOMINANT_VARIABLES_SHOWN = 3


def _variance_table(context: ViewContext):
    variance = context.analysis.pca.variance.rename(
        columns={
            pca.EIGENVALUE: "Autovalor",
            pca.VARIANCE_RATIO: "Varianza explicada",
            pca.CUMULATIVE_RATIO: "Varianza acumulada",
        }
    )
    return variance.style.format(
        {"Autovalor": "{:.2f}", "Varianza explicada": "{:.1%}", "Varianza acumulada": "{:.1%}"}
    )


def _describe_component(context: ViewContext, component: str) -> str:
    result = context.analysis.pca
    dominant = insights.dominant_variables(result.loadings, component, DOMINANT_VARIABLES_SHOWN)
    described = ", ".join(
        f"{charts.label(variable)} ({loading:+.2f})" for variable, loading in dominant.items()
    )
    ratio = result.variance.loc[component, pca.VARIANCE_RATIO]
    return f"- **{component}** ({ratio:.1%} de la varianza): {described}."


def _justify_components(context: ViewContext) -> str:
    analysis = context.analysis
    kaiser = pca.components_by_kaiser(analysis.pca)
    rule = analysis.settings.component_rule
    reasons = {
        ComponentRule.KAISER: "son los que tienen autovalor mayor que 1, es decir, los que "
        "explican más que una variable original por sí sola",
        ComponentRule.CUMULATIVE_VARIANCE: "son los mínimos necesarios para alcanzar el "
        f"{analysis.settings.variance_threshold:.0%} de varianza acumulada",
        ComponentRule.MANUAL: "es una elección manual, útil para ver qué tan sensibles son los "
        f"segmentos a esta decisión (el criterio de Kaiser sugeriría {kaiser})",
    }
    return (
        f"Se conservan **{analysis.n_components} componentes**, que retienen el "
        f"**{analysis.retained_variance:.1%}** de la variación original: {reasons[rule]}."
    )


def render(context: ViewContext) -> None:
    analysis = context.analysis
    result = analysis.pca

    st.title("Análisis de Componentes Principales")
    st.markdown(
        "El ACP construye nuevas variables (componentes) que son combinaciones de las "
        "dimensiones estandarizadas, no están correlacionadas entre sí y se ordenan de mayor a "
        "menor varianza explicada."
    )

    st.subheader("Varianza explicada")
    table, plot = st.columns([2, 3])
    table.dataframe(_variance_table(context), width="stretch")
    plot.plotly_chart(charts.scree_plot(result.variance, analysis.n_components), width="stretch")
    st.info(_justify_components(context))
    st.markdown(
        """
        **Cómo leerlo.** Si las dimensiones midieran un único fenómeno, el primer componente
        concentraría casi toda la varianza y la curva caería de golpe. Una caída gradual, como la
        que suele aparecer aquí, indica que el desempeño municipal **no se reduce a una o dos
        dimensiones**: resumirlo en un plano implica perder una parte importante de la información.
        """
    )

    st.subheader("Cargas: qué significa cada componente")
    st.plotly_chart(charts.loadings_heatmap(result.loadings), width="stretch")
    retained = result.components[: analysis.n_components]
    st.markdown(
        "Cada carga es la correlación entre una dimensión y un componente. Variables con mayor "
        "peso en los componentes conservados:\n\n"
        + "\n".join(_describe_component(context, component) for component in retained)
    )
    st.caption(
        "El signo indica la dirección: cargas de signo contrario en un mismo componente señalan "
        "dimensiones que tienden a moverse en sentidos opuestos."
    )

    st.subheader("Municipios en el espacio de componentes")
    left, right, toggle = st.columns(3)
    x = left.selectbox("Eje horizontal", result.components, index=0)
    y = right.selectbox("Eje vertical", result.components, index=1)
    show_loadings = toggle.toggle("Mostrar cargas (biplot)", value=True)

    figure = charts.component_scatter(result, (x, y), analysis.clean[config.MUNICIPALITY])
    if show_loadings:
        charts.add_loading_vectors(figure, result, (x, y))
    charts.add_highlight(figure, result, (x, y), context.highlight)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Cada punto es un municipio. Las flechas muestran hacia dónde se desplaza un municipio "
        "cuando mejora en esa dimensión. El título indica qué porcentaje de la información "
        "original resume este plano; el resto queda en los demás componentes."
    )
