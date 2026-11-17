"""Figuras de Plotly. Cada función recibe tablas ya calculadas y devuelve una figura."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mdm import clustering, config, pca

SEGMENT_COLORS = px.colors.qualitative.Safe
PRIMARY = "#1F4E79"
ACCENT = "#C8553D"
MUTED = "#9AA5B1"
DIVERGING_SCALE = "RdBu"
HIGHLIGHT_SYMBOL = "star"
LOADING_VECTOR_REACH = 0.9  # fracción del rango de puntajes que ocupan las flechas


def label(column: str) -> str:
    """Nombre legible de una variable del modelo."""
    return config.DIMENSION_LABELS.get(column, column)


def _relabel(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(index=label, columns=label)


def _segment_color(segments: list[str]) -> dict[str, str]:
    return {segment: SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i, segment in enumerate(segments)}


def _heatmap(table: pd.DataFrame, title: str, value_format: str, limit: float) -> go.Figure:
    figure = px.imshow(
        table,
        text_auto=value_format,
        color_continuous_scale=DIVERGING_SCALE,
        zmin=-limit,
        zmax=limit,
        aspect="auto",
        title=title,
    )
    figure.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar_title=None)
    return figure


def distribution_boxes(frame: pd.DataFrame, variables: list[str]) -> go.Figure:
    """Diagramas de caja de los puntajes originales, todos en escala 0-100."""
    long = _relabel(frame[variables]).melt(var_name="Dimensión", value_name="Puntaje")
    figure = px.box(
        long,
        x="Puntaje",
        y="Dimensión",
        color_discrete_sequence=[PRIMARY],
        title="Distribución de los puntajes por dimensión",
    )
    figure.update_layout(yaxis_title=None, xaxis_title="Puntaje (0 a 100)")
    return figure


def correlation_heatmap(frame: pd.DataFrame, variables: list[str]) -> go.Figure:
    """Matriz de correlaciones de Pearson entre las variables seleccionadas."""
    correlations = _relabel(frame[variables].corr())
    return _heatmap(correlations, "Correlaciones entre dimensiones", ".2f", limit=1.0)


def scree_plot(variance: pd.DataFrame, n_components: int) -> go.Figure:
    """Gráfico de sedimentación: varianza por componente y su acumulado."""
    retained = [ACCENT if i < n_components else MUTED for i in range(len(variance))]
    figure = go.Figure()
    figure.add_bar(
        x=variance.index,
        y=variance[pca.VARIANCE_RATIO],
        marker_color=retained,
        name="Varianza explicada",
    )
    figure.add_scatter(
        x=variance.index,
        y=variance[pca.CUMULATIVE_RATIO],
        mode="lines+markers",
        line_color=PRIMARY,
        name="Varianza acumulada",
    )
    figure.update_layout(
        title="Sedimentación: varianza explicada por componente",
        xaxis_title="Componente principal",
        yaxis_title="Proporción de la varianza total",
        yaxis_tickformat=".0%",
        legend_orientation="h",
        legend_y=-0.2,
    )
    return figure


def loadings_heatmap(loadings: pd.DataFrame) -> go.Figure:
    """Cargas: correlación entre cada variable original y cada componente."""
    table = loadings.rename(index=label)
    return _heatmap(table, "Cargas de las variables en cada componente", ".2f", limit=1.0)


def component_scatter(
    result: pca.PcaResult,
    axes: tuple[str, str],
    hover_name: pd.Series,
    color: pd.Series | None = None,
) -> go.Figure:
    """Observaciones en el plano de dos componentes, opcionalmente por segmento."""
    x, y = axes
    ratios = result.variance[pca.VARIANCE_RATIO]
    points = result.scores.assign(**{config.MUNICIPALITY: hover_name})
    segments: list[str] = []
    if color is not None:
        points[config.CLUSTER] = color
        segments = sorted(color.unique())
    figure = px.scatter(
        points,
        x=x,
        y=y,
        color=config.CLUSTER if segments else None,
        hover_name=config.MUNICIPALITY,
        category_orders={config.CLUSTER: segments},
        color_discrete_map=_segment_color(segments),
        color_discrete_sequence=[MUTED],
        opacity=0.7,
        title=f"Municipios en el plano {x}-{y} ({ratios[x] + ratios[y]:.1%} de la varianza total)",
        labels={x: f"{x} ({ratios[x]:.1%})", y: f"{y} ({ratios[y]:.1%})"},
    )
    figure.update_layout(legend_title="Segmento" if segments else None)
    return figure


def add_loading_vectors(figure: go.Figure, result: pca.PcaResult, axes: tuple[str, str]) -> None:
    """Superpone las cargas como flechas para leer el plano como un biplot."""
    x, y = axes
    reach = float(result.scores[[x, y]].abs().max().max()) * LOADING_VECTOR_REACH
    for variable, row in result.loadings.iterrows():
        tip = {"x": row[x] * reach, "y": row[y] * reach}
        figure.add_annotation(
            **tip,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowwidth=1.5,
            arrowcolor=PRIMARY,
        )
        figure.add_annotation(
            **tip, text=label(str(variable)), showarrow=False, font_color=PRIMARY, yshift=10
        )


def add_highlight(
    figure: go.Figure, result: pca.PcaResult, axes: tuple[str, str], municipality: pd.Series
) -> None:
    """Marca con una estrella un municipio; ``municipality`` es su fila de datos."""
    x, y = axes
    position = result.scores.loc[municipality.name]
    figure.add_scatter(
        x=[position[x]],
        y=[position[y]],
        mode="markers+text",
        text=[municipality[config.MUNICIPALITY]],
        textposition="top center",
        marker={"symbol": HIGHLIGHT_SYMBOL, "size": 16, "color": "black"},
        name=municipality[config.MUNICIPALITY],
        hoverinfo="skip",
    )


def _k_curve(
    evaluation: pd.DataFrame, metric: str, chosen_k: int, title: str, axis: str
) -> go.Figure:
    figure = px.line(evaluation, y=metric, markers=True, title=title)
    figure.update_traces(line_color=PRIMARY)
    figure.add_vline(
        x=chosen_k, line_dash="dash", line_color=ACCENT, annotation_text=f"K = {chosen_k}"
    )
    figure.update_layout(xaxis_title="Número de segmentos (K)", yaxis_title=axis, xaxis_dtick=1)
    return figure


def elbow_plot(evaluation: pd.DataFrame, chosen_k: int) -> go.Figure:
    """Método del codo: inercia intra-clúster para cada K."""
    return _k_curve(
        evaluation, clustering.INERTIA, chosen_k,
        "Método del codo", "Inercia (suma de distancias² al centroide)",
    )  # fmt: skip


def silhouette_by_k_plot(evaluation: pd.DataFrame, chosen_k: int) -> go.Figure:
    """Coeficiente de silueta medio para cada K."""
    return _k_curve(
        evaluation, clustering.SILHOUETTE, chosen_k,
        "Coeficiente de silueta", "Silueta media",
    )  # fmt: skip


def silhouette_plot(result: clustering.ClusteringResult) -> go.Figure:
    """Silueta de cada municipio, ordenada dentro de su segmento."""
    table = pd.DataFrame({config.CLUSTER: result.labels, "silueta": result.sample_silhouettes})
    table = table.sort_values([config.CLUSTER, "silueta"], ascending=[True, False])
    table["posición"] = range(len(table))
    figure = px.bar(
        table,
        x="posición",
        y="silueta",
        color=config.CLUSTER,
        color_discrete_map=_segment_color(result.segments),
        title="Silueta por municipio dentro de cada segmento",
    )
    figure.add_hline(
        y=result.silhouette,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Media = {result.silhouette:.2f}",
    )
    figure.update_traces(marker_line_width=0)
    figure.update_layout(
        bargap=0,
        xaxis_title="Municipios (ordenados por segmento y silueta)",
        yaxis_title="Coeficiente de silueta",
        xaxis_showticklabels=False,
        legend_title="Segmento",
    )
    return figure


def segment_profile_heatmap(standardized_profile: pd.DataFrame) -> go.Figure:
    """Perfil de cada segmento en desviaciones estándar frente al promedio nacional."""
    table = standardized_profile.rename(columns=label)
    table.index = [f"Segmento {segment}" for segment in table.index]
    limit = float(table.abs().max().max())
    return _heatmap(
        table, "Perfil de los segmentos (desviaciones estándar frente al promedio)", ".2f", limit
    )


def segment_map(frame: pd.DataFrame, labels: pd.Series) -> go.Figure:
    """Ubicación geográfica de los municipios, coloreados por segmento."""
    segments = sorted(labels.unique())
    figure = px.scatter_map(
        frame.assign(**{config.CLUSTER: labels}),
        lat=config.LATITUDE,
        lon=config.LONGITUDE,
        color=config.CLUSTER,
        hover_name=config.MUNICIPALITY,
        hover_data={config.DEPARTMENT: True, config.LATITUDE: False, config.LONGITUDE: False},
        category_orders={config.CLUSTER: segments},
        color_discrete_map=_segment_color(segments),
        zoom=4.2,
        height=620,
        map_style="carto-positron",
        title="Distribución geográfica de los segmentos",
    )
    figure.update_layout(legend_title="Segmento")
    return figure


def crosstab_heatmap(crosstab: pd.DataFrame) -> go.Figure:
    """Porcentaje de cada grupo del DNP que cae en cada segmento."""
    figure = px.imshow(
        crosstab,
        text_auto=".0f",
        color_continuous_scale="Blues",
        zmin=0,
        zmax=100,
        aspect="auto",
        title="Segmentos frente a los grupos de capacidades iniciales del DNP (% por fila)",
    )
    figure.update_layout(xaxis_title="Segmento", yaxis_title=None, coloraxis_colorbar_title="%")
    return figure
