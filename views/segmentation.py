"""Pantalla 5: segmentación con K-Means."""

from __future__ import annotations

import streamlit as st

from mdm import charts, clustering, config, insights
from views.context import ViewContext

DISTINCTIVE_VARIABLES_SHOWN = 3


def _justify_k(context: ViewContext) -> str:
    analysis = context.analysis
    silhouettes = analysis.k_evaluation[clustering.SILHOUETTE]
    chosen, best = silhouettes[analysis.k], silhouettes[analysis.suggested_k]
    if analysis.k == analysis.suggested_k:
        return (
            f"Se usa **K = {analysis.k}**, el valor con la mayor silueta media ({chosen:.3f}). "
            "El codo se revisa como apoyo: la inercia siempre baja al aumentar K, así que solo "
            "importa dónde deja de bajar con fuerza."
        )
    return (
        f"Se usa **K = {analysis.k}** por elección manual (silueta {chosen:.3f}). La silueta "
        f"máxima corresponde a K = {analysis.suggested_k} ({best:.3f}): la diferencia de "
        f"{best - chosen:.3f} es el costo en separación que se paga por tener segmentos más "
        "detallados."
    )


def _describe_segment(context: ViewContext, segment: str) -> str:
    analysis = context.analysis
    size = int(analysis.segment_profile.loc[segment, clustering.SIZE])
    distinctive = insights.distinctive_variables(
        analysis.standardized_profile, segment, DISTINCTIVE_VARIABLES_SHOWN
    )
    described = ", ".join(
        f"{charts.label(variable)} ({deviation:+.2f} σ)"
        for variable, deviation in distinctive.items()
    )
    share = size / len(analysis.clean)
    return f"- **Segmento {segment}** ({size} municipios, {share:.0%}): {described}."


def _profile_table(context: ViewContext):
    profile = context.analysis.segment_profile.rename(columns=charts.label)
    profile.index = [f"Segmento {segment}" for segment in profile.index]
    formats = dict.fromkeys(profile.columns, "{:.1f}") | {clustering.SIZE: "{:.0f}"}
    return profile.style.format(formats)


def render(context: ViewContext) -> None:
    analysis = context.analysis
    result = analysis.clustering
    labels = result.labels
    highlight_name = context.highlight[config.MUNICIPALITY]

    st.title("Segmentación con K-Means")
    st.markdown(
        f"K-Means se aplica sobre los puntajes de los **{analysis.n_components} componentes "
        f"conservados** ({analysis.retained_variance:.1%} de la varianza). Así se agrupa con la "
        "estructura principal de los datos y se evita que los componentes de menor varianza "
        "pesen en las distancias."
    )

    st.subheader("¿Cuántos segmentos?")
    elbow, silhouette = st.columns(2)
    elbow.plotly_chart(charts.elbow_plot(analysis.k_evaluation, analysis.k), width="stretch")
    silhouette.plotly_chart(
        charts.silhouette_by_k_plot(analysis.k_evaluation, analysis.k), width="stretch"
    )
    st.info(_justify_k(context))

    st.subheader("Calidad de la separación")
    doubtful = insights.share_below(result.sample_silhouettes, 0)
    first, second, third = st.columns(3)
    first.metric("Silueta media", f"{result.silhouette:.3f}")
    second.metric("Inercia", f"{result.inertia:,.0f}")
    third.metric("Municipios con silueta negativa", f"{doubtful:.1%}")
    st.plotly_chart(charts.silhouette_plot(result), width="stretch")
    st.markdown(
        """
        **Cómo leerlo.** La silueta va de −1 a 1. Cerca de 1, el municipio está claramente dentro
        de su segmento; cerca de 0, está en la frontera entre dos; negativa, se parece más a otro
        segmento que al asignado. Como referencia habitual (Kaufman y Rousseeuw), una media entre
        0,26 y 0,50 señala una estructura débil y por debajo de 0,25 no hay estructura sustancial:
        los segmentos son **regiones de un continuo** más que grupos nítidamente separados.
        """
    )

    st.subheader("Segmentos en el espacio de componentes")
    left, right = st.columns(2)
    x = left.selectbox("Eje horizontal", analysis.pca.components, index=0, key="segment_x")
    y = right.selectbox("Eje vertical", analysis.pca.components, index=1, key="segment_y")
    figure = charts.component_scatter(
        analysis.pca, (x, y), analysis.clean[config.MUNICIPALITY], labels
    )
    charts.add_highlight(figure, analysis.pca, (x, y), context.highlight)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{highlight_name} pertenece al **segmento {labels[context.highlight.name]}** "
        f"(silueta {result.sample_silhouettes[context.highlight.name]:.2f})."
    )

    st.subheader("¿Qué caracteriza a cada segmento?")
    st.markdown("Promedios en la escala original (0–100):")
    st.dataframe(_profile_table(context), width="stretch")
    st.plotly_chart(charts.segment_profile_heatmap(analysis.standardized_profile), width="stretch")
    st.markdown(
        "Dimensiones en las que cada segmento se aleja más del promedio nacional "
        "(en desviaciones estándar, σ):\n\n"
        + "\n".join(_describe_segment(context, segment) for segment in result.segments)
    )
    st.caption(
        "Las letras son solo nombres: se asignan del segmento más grande al más pequeño y no "
        "implican que un segmento sea «mejor» que otro."
    )

    st.subheader("¿Dónde están?")
    st.plotly_chart(charts.segment_map(analysis.clean, labels), width="stretch")
    st.caption(
        "Los puntos forman la silueta del país. El punto aislado arriba a la izquierda es "
        "Providencia, en el archipiélago de San Andrés."
    )

    st.subheader("Contraste con los grupos del DNP")
    crosstab = clustering.cross_tabulate(labels, analysis.clean[config.PEER_GROUP])
    st.plotly_chart(charts.crosstab_heatmap(crosstab), width="stretch")
    st.caption(
        "El grupo de capacidades iniciales no entró al modelo. Si los segmentos se alinean con "
        "él, el desempeño observado depende en buena medida del punto de partida del municipio."
    )

    st.download_button(
        "Descargar municipios con su segmento (CSV)",
        analysis.clean.assign(**{config.CLUSTER: labels}).to_csv().encode("utf-8"),
        file_name=f"mdm_segmentos_{analysis.settings.year}.csv",
        mime="text/csv",
    )
