"""Pantalla 6: discusión de resultados, generada a partir de la corrida actual."""

from __future__ import annotations

import streamlit as st

from mdm import charts, clustering, config, insights, pca
from views.context import ViewContext

VARIABLES_PER_FINDING = 3
WEAK_STRUCTURE = 0.25
REASONABLE_STRUCTURE = 0.50


def _signed_list(values) -> str:
    return ", ".join(f"{charts.label(name)} ({value:+.2f})" for name, value in values.items())


def _components_answer(context: ViewContext) -> str:
    analysis = context.analysis
    lines = []
    for component in analysis.pca.components[: analysis.n_components]:
        dominant = insights.dominant_variables(
            analysis.pca.loadings, component, VARIABLES_PER_FINDING
        )
        ratio = analysis.pca.variance.loc[component, pca.VARIANCE_RATIO]
        lines.append(f"- **{component}** ({ratio:.1%}): {_signed_list(dominant)}.")
    return "\n".join(lines)


def _separation_verdict(silhouette: float) -> str:
    if silhouette < WEAK_STRUCTURE:
        return (
            "no hay una estructura de grupos sustancial: los segmentos son cortes sobre un "
            "continuo y deben leerse como tipologías orientativas"
        )
    if silhouette < REASONABLE_STRUCTURE:
        return (
            "la estructura es débil: existen perfiles reconocibles, pero con fronteras difusas "
            "y muchos municipios intermedios"
        )
    return "los grupos están razonablemente separados"


def _segments_answer(context: ViewContext) -> str:
    analysis = context.analysis
    lines = []
    for segment in analysis.clustering.segments:
        size = int(analysis.segment_profile.loc[segment, clustering.SIZE])
        distinctive = insights.distinctive_variables(
            analysis.standardized_profile, segment, VARIABLES_PER_FINDING
        )
        lines.append(
            f"- **Segmento {segment}** ({size} municipios): {_signed_list(distinctive)} σ."
        )
    return "\n".join(lines)


def _peer_groups_answer(context: ViewContext) -> str:
    analysis = context.analysis
    crosstab = clustering.cross_tabulate(
        analysis.clustering.labels, analysis.clean[config.PEER_GROUP]
    )
    lines = [
        f"- **{group}**: {shares.max():.0f}% en el segmento {shares.idxmax()}."
        for group, shares in crosstab.iterrows()
    ]
    return "\n".join(lines)


def render(context: ViewContext) -> None:
    analysis = context.analysis
    result = analysis.clustering
    doubtful = insights.share_below(result.sample_silhouettes, 0)

    st.title("Discusión de resultados")
    st.caption(
        f"Corrida actual: año {analysis.settings.year}, {len(analysis.variables)} dimensiones, "
        f"{analysis.n_components} componentes, K = {analysis.k}. "
        "Esta página se reescribe cuando cambias las decisiones del panel lateral."
    )

    st.subheader("1. ¿Qué variables definen los componentes principales?")
    st.markdown(_components_answer(context))
    st.markdown(
        "Los valores entre paréntesis son cargas (correlación con el componente). El primer "
        "componente recoge lo que comparten las dimensiones más correlacionadas; los siguientes "
        "capturan las dimensiones que se mueven por su cuenta."
    )

    st.subheader("2. ¿Qué proporción de la variación original se conservó?")
    st.markdown(
        f"**{analysis.retained_variance:.1%}** con {analysis.n_components} de "
        f"{len(analysis.variables)} componentes. El {1 - analysis.retained_variance:.1%} restante "
        "queda fuera de la segmentación: es el precio de simplificar."
    )

    st.subheader("3. ¿Qué diferencias reales aparecen entre los grupos?")
    st.markdown(_segments_answer(context))
    st.markdown(
        "Los valores están en desviaciones estándar frente al promedio nacional: positivos "
        "indican mejor puntaje que el promedio en esa dimensión; negativos, peor."
    )

    st.subheader("4. ¿Son grupos claramente distinguibles?")
    verdict = _separation_verdict(result.silhouette)
    st.markdown(
        f"La silueta media es **{result.silhouette:.3f}** y el **{doubtful:.1%}** de los "
        f"municipios tiene silueta negativa. Con ese valor, {verdict}."
    )

    st.subheader("5. ¿Qué dice esto sobre la pregunta inicial?")
    st.markdown(
        "Proporción de cada grupo de capacidades iniciales del DNP que cae en su segmento más "
        "frecuente:\n\n" + _peer_groups_answer(context)
    )
    st.markdown(
        """
        Si los grupos de mayor capacidad inicial se concentran en un segmento y los de menor
        capacidad en otro, el perfil de desempeño está fuertemente condicionado por el punto de
        partida: la segmentación *redescubre* la desigualdad territorial más que diferencias de
        gestión. Los grupos intermedios repartidos entre varios segmentos son los más
        interesantes para la política pública, porque allí municipios con capacidades similares
        obtienen perfiles distintos.
        """
    )

    st.subheader("Limitaciones")
    st.markdown(
        """
        - **Segmentos poco nítidos.** Con siluetas bajas, un municipio cercano a la frontera
          podría cambiar de segmento con otro año, otra semilla u otro K. No debe usarse la
          etiqueta para decisiones individuales sin revisar su silueta.
        - **K-Means supone grupos compactos y de tamaño similar**, y es sensible a valores
          extremos como los de Movilización de recursos, muy asimétrica.
        - **El ACP solo captura relaciones lineales** y los componentes dependen de las
          dimensiones incluidas: quitar una variable cambia su significado.
        - **Las dimensiones son índices compuestos del DNP** basados en registros administrativos;
          la calidad del reporte de cada municipio afecta su puntaje.
        - **Es un corte transversal:** describe perfiles en un año, no trayectorias ni causas.
        """
    )
