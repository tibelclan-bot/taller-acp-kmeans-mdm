"""Pantalla 1: contexto, pregunta, fuente y pertinencia de los métodos."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mdm import config
from views.context import ViewContext, get_panel


def _variables_table() -> pd.DataFrame:
    rows = [
        {"Dimensión": label, "Componente del DNP": component, "Columna": column}
        for column, (label, component) in config.DIMENSIONS.items()
    ]
    return pd.DataFrame(rows)


def render(context: ViewContext) -> None:
    panel = get_panel()
    municipalities = panel[config.MUNICIPALITY_CODE].nunique()
    departments = panel[config.DEPARTMENT_CODE].nunique()
    first_year, last_year = panel[config.YEAR].min(), panel[config.YEAR].max()

    st.title("Perfiles de desempeño municipal en Colombia")
    st.caption(f"Taller individual: ACP y K-Means · {config.AUTHOR} · {config.COURSE}")

    st.header("La pregunta")
    st.markdown(
        """
        El Departamento Nacional de Planeación (DNP) resume el desempeño de cada alcaldía en un
        solo número: la **Medición del Desempeño Municipal (MDM)**. Ese número es útil para hacer
        un ranking, pero esconde *cómo* se llega a él: dos municipios con el mismo puntaje pueden
        tener perfiles opuestos, por ejemplo uno que recauda bien pero es inseguro y otro que
        depende de transferencias pero tiene buena cobertura en salud.

        Este proyecto pregunta:

        > **¿Qué perfiles de desempeño existen entre los municipios colombianos y qué tanto
        > coinciden con los grupos de capacidades iniciales con los que el DNP los compara?**

        Responderla sirve para diseñar asistencia técnica diferenciada: no es lo mismo apoyar a un
        municipio que falla en transparencia que a uno que falla en recaudo.
        """
    )

    st.header("Los datos")
    left, middle, right = st.columns(3)
    left.metric("Municipios", f"{municipalities:,}".replace(",", "."))
    middle.metric("Departamentos (incluye Bogotá)", departments)
    right.metric("Años disponibles", f"{first_year}–{last_year}")
    st.markdown(
        f"""
        - **Fuente:** DNP, Medición del Desempeño Municipal, vigencias {first_year}–{last_year}.
          Es información pública y agregada por municipio: no contiene datos personales.
        - **Unidad de observación:** un municipio en un año. El análisis usa un corte transversal
          (por defecto {last_year}); el año se cambia en el panel lateral.
        - **Variables del modelo:** las ocho dimensiones de la MDM, cada una en escala de 0 a 100,
          donde un puntaje mayor indica mejor desempeño.
        - **Variables que no entran al modelo:** códigos DANE y nombres (identificadores),
          coordenadas (solo para el mapa), el índice MDM y sus componentes de Gestión y Resultados
          (son combinaciones de las ocho dimensiones, así que duplicarían información) y el grupo
          de capacidades iniciales (se reserva para contrastar los segmentos al final).
        """
    )
    st.dataframe(_variables_table(), hide_index=True, width="stretch")

    st.header("Limitaciones de la fuente")
    st.markdown(
        """
        - Cada dimensión ya es un **índice compuesto**: resume varios indicadores con las
          ponderaciones del DNP, así que el análisis hereda esas decisiones.
        - Buena parte de los insumos son **registros administrativos reportados por las propias
          entidades**; un puntaje bajo puede reflejar mala gestión o simplemente mal reporte.
        - El archivo del proyecto se construyó a partir de una **copia pública de la base del
          DNP**; antes de usar las cifras para decisiones reales conviene contrastarlas con la
          publicación oficial.
        - Es una **fotografía de un año**: no dice si un municipio mejora o empeora.
        """
    )

    st.header("¿Por qué ACP y K-Means?")
    st.markdown(
        """
        - **ACP.** Las ocho dimensiones están correlacionadas entre sí (un municipio con buen
          recaudo suele tener mejor ordenamiento territorial y servicios). El ACP permite ver
          cuántas dimensiones *independientes* hay realmente detrás de la tabla y qué variables
          las definen, en lugar de aceptar de entrada la división Gestión/Resultados del DNP.
        - **K-Means.** El DNP agrupa los municipios *antes* de medirlos, según su punto de
          partida. K-Means hace lo contrario: los agrupa *según lo que obtuvieron*. Comparar
          ambas clasificaciones muestra si el desempeño está atado a las capacidades iniciales.
        - Ambos métodos se basan en varianzas y distancias, por eso las variables se
          **estandarizan** antes de aplicarlos.
        """
    )
    st.info(
        "Recorre las pantallas en orden desde el menú lateral. Todas se recalculan con las "
        "decisiones que tomes en el panel **Decisiones del análisis**."
    )
