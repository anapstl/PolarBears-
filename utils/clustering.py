import streamlit as st, pandas as pd, pickle, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def mostrar(df):
    st.markdown("## Clustering interactivo")
    variables_cluster = [
        "distance_per_day", "bearing_change", "velocity_kmh", "acceleration_kmph2"
    ]
    df_cluster = df.dropna(subset=variables_cluster).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[variables_cluster])

    k = st.sidebar.slider("Selecciona número de clusters (k)", 2, 8, 3)

    with st.spinner("Agrupando osas polares..."):
        kmeans = KMeans(n_clusters=k, random_state=10)
        df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_cluster["pca_1"] = X_pca[:, 0]
        df_cluster["pca_2"] = X_pca[:, 1]

        resumen = df_cluster.groupby("cluster")[variables_cluster].mean().round(2)

        # Asignación de nombres automáticos a clusters
        nombres_disponibles = [
            "The Wanderers", "Direction Shifters", "Beaufort Trackers",
            "Ice Chasers", "Arctic Strollers", "Zigzag Nomads",
            "Steady Striders", "Polar Explorers"
        ]

        orden = resumen["distance_per_day"].sort_values(ascending=False).index
        nombres_ordenados = {cluster_id: nombre for cluster_id, nombre in zip(orden, nombres_disponibles)}
        # Paleta de colores consistente
        palette = sns.color_palette("Set2", n_colors=k)  # o "tab10", "Set3", etc.
        colores_por_nombre = {nombre: to_hex(color) for nombre, color in zip(nombres_ordenados.values(), palette)}

        df_cluster["cluster_name"] = df_cluster["cluster"].map(nombres_ordenados)
        resumen["cluster_name"] = resumen.index.map(nombres_ordenados)

        # Visualización PCA
        # st.subheader("Visualización PCA")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        sns.scatterplot(data=df_cluster, x="pca_1", y="pca_2", hue="cluster_name", palette=colores_por_nombre, ax=ax)
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # tu leyenda actual
        legend.get_frame().set_alpha(0.0)  # hace el fondo completamente transparente
        ax.set_title("Visualización PCA")
        st.pyplot(fig, transparent=True)

        # Radar plot
        st.subheader("Radar por cluster")
        fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor="none")
        categories = variables_cluster
        N = len(categories)

        for cluster_id, row in resumen.iterrows():
            values = row[variables_cluster].values.tolist()
            values += values[:1]  # cerrar el radar
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            color = colores_por_nombre.get(row["cluster_name"], "#333333")  # color por cluster
            ax_radar.plot(angles, values, label=row["cluster_name"], color=color)
            ax_radar.fill(angles, values, alpha=0.1, color=color)

        ax_radar.set_xticks([n / float(N) * 2 * np.pi for n in range(N)])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_title("Resumen de patrones por cluster")
        legend = ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        legend.get_frame().set_alpha(0.0)
        st.pyplot(fig_radar, transparent=True)

        # st.subheader("Resumen estadístico por cluster")
        # st.dataframe(resumen.set_index("cluster_name"))

        st.subheader("Resumen por grupo (cluster)")
        st.markdown("""
        Cada fila representa un grupo de comportamiento identificado entre las osas polares. Las métricas indican su movimiento medio:

        - `distance_per_day`: km recorridos por día  
        - `bearing_change`: cambios de dirección promedio (grados)  
        - `velocity_kmh`: velocidad media (km/h)  
        - `acceleration_kmph2`: aceleración media (km/h²)
        """)
        st.dataframe(resumen.style.format(precision=2))