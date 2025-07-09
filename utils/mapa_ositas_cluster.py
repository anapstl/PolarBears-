import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def mostrar_ositas_cluster_pydeck(cluster_summary):
    """
    Visualiza clusters de osas polares usando PyDeck.
    Cada punto representa el centro geográfico de un cluster,
    escalado por cantidad de osas y coloreado por ID.
    """

    # Asegurar columnas en minúsculas
    df = cluster_summary.copy()
    df.columns = df.columns.str.lower()

    # Generar colores magma
    cmap = cm.get_cmap("Set1", len(df))
    norm = mcolors.Normalize(vmin=0, vmax=len(df) - 1)
    df["color"] = [
        [int(r * 255), int(g * 255), int(b * 255), 160]
        for r, g, b, _ in cmap(norm(df.index))
    ]

    # Escalar radios entre 5 km y 25 km
    min_count = df["uniqueanimalid"].min()
    max_count = df["uniqueanimalid"].max()
    df["radius"] = np.interp(df["uniqueanimalid"], (min_count, max_count), (10000, 40000))

    # Crear capa Scatterplot
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[avglongitude, avglatitude]',
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )

    # Vista inicial centrada
    view_state = pdk.ViewState(
        latitude=df["avglatitude"].mean(),
        longitude=df["avglongitude"].mean(),
        zoom=2,
        pitch=0,
    )

    # Crear mapa
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "🐾 Cluster {cluster}\nOsas: {uniqueanimalid}"}
    )

    return deck
