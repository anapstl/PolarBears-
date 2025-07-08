# utils/geo_clustering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
from ipyleaflet import Map, CircleMarker, LayerGroup, WidgetControl
import ipywidgets as widgets
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import base64
from utils.mapa_ositas import crear_leyenda_magma, mostrar_ositas_cluster


def mostrar(df):
    st.markdown("## Agrupación geográfica de osas polares")
    st.markdown("Selecciona el número de clusters para explorar cómo se agrupan según su posición promedio y distancia total.")

    # 1. Agrupar por osa
    # df_individuos = df.groupby("UniqueAnimalID").agg({
    #     "AvgLatitude": "first",
    #     "AvgLongitude": "first",
    #     "TotalDistance_km": "first"
    # }).reset_index()

    # df_individuos.rename(columns={"TotalDistance_km": "TotalDistance"}, inplace=True)

    # Variables seleccionadas
    variables_geo = ["AvgLatitude", "AvgLongitude", "TotalDistance_km"]
    df_geo = df.dropna(subset=variables_geo).copy()

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_geo[variables_geo])

    # Slider
    k = st.sidebar.slider("Número de clusters (k)", 2, 8, 3)

    with st.spinner("Calculando agrupaciones..."):
        kmeans = KMeans(n_clusters=k, random_state=10)
        df_geo["cluster"] = kmeans.fit_predict(X_scaled)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_geo["pca_1"] = X_pca[:, 0]
        df_geo["pca_2"] = X_pca[:, 1]

        resumen = df_geo.groupby("cluster")[variables_geo].mean().round(2)

        # Nombres automáticos
        nombres_disponibles = [
            "Beaufort Explorers", "Chukchi Roamers", "Arctic Anchored",
            "Nomadic Drifters", "Western Trackers", "Northern Scouts",
            "Coastal Cruisers", "Ice Followers"
        ]
        orden = resumen["TotalDistance_km"].sort_values(ascending=False).index
        nombres_ordenados = {cluster_id: nombre for cluster_id, nombre in zip(orden, nombres_disponibles)}
        palette = sns.color_palette("Set2", n_colors=k)
        colores_por_nombre = {nombre: to_hex(color) for nombre, color in zip(nombres_ordenados.values(), palette)}

        df_geo["cluster_name"] = df_geo["cluster"].map(nombres_ordenados)
        resumen["cluster_name"] = resumen.index.map(nombres_ordenados)

        # Gráfico PCA
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        sns.scatterplot(data=df_geo, x="pca_1", y="pca_2", hue="cluster_name", palette=colores_por_nombre, ax=ax)
        ax.set_title("Distribución PCA de osas por cluster geográfico")
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        legend.get_frame().set_alpha(0.0)
        st.pyplot(fig, transparent=True)

        # Radar
        st.subheader("Radar por grupo geográfico")
        fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="none")
        categories = variables_geo
        N = len(categories)

        for cluster_id, row in resumen.iterrows():
            values = row[variables_geo].values.tolist() + [row[variables_geo[0]]]
            angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
            color = colores_por_nombre.get(row["cluster_name"], "#444444")
            ax_radar.plot(angles, values, label=row["cluster_name"], color=color)
            ax_radar.fill(angles, values, alpha=0.1, color=color)

        ax_radar.set_xticks([n / float(N) * 2 * np.pi for n in range(N)])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_title("Resumen de patrones geográficos")
        legend = ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        legend.get_frame().set_alpha(0.0)
        st.pyplot(fig_radar, transparent=True)

        st.subheader("Resumen por grupo")
        st.markdown("""
        - `AvgLatitude`: Latitud promedio de seguimiento  
        - `AvgLongitude`: Longitud promedio  
        - `TotalDistance_km`: Distancia total recorrida
        """)
        st.dataframe(resumen.set_index("cluster_name").style.format(precision=2))

        #         # Mapa interactivo con ipyleaflet
        # st.subheader("Visualización geográfica en mapa")

        # mostrar_mapa = st.checkbox("Mostrar mapa interactivo de clusters")

        # if mostrar_mapa:
        #     # Preparamos el DataFrame resumen con número de osas por cluster
        #     resumen_geo = (
        #         df_geo.groupby("cluster")
        #         .agg({
        #             "AvgLatitude": "mean",
        #             "AvgLongitude": "mean",
        #             "UniqueAnimalID": "count"
        #         })
        #         .rename(columns={"UniqueAnimalID": "uniqueanimalid"})
        #         .reset_index()
        #     )

        #     st.markdown("Haz zoom y desplázate para explorar los grupos.")
        #     st.components.v1.html(
        #         mostrar_ositas_cluster(resumen_geo)._repr_html_(),
        #         height=500,
        #         scrolling=True
        #     )

