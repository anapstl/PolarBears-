import base64
import streamlit as st
import pandas as pd
from utils import regresion, clasificacion
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import clustering  # si lo modularizas luego

def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stAppHeader {{
        display: none;
    }}
    .stSidebar {{
        background-color: rgba(240, 242, 246, 0.62);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Configuración inicial
st.set_page_config(page_title="Exploradores Polares", layout="wide")

st.markdown("""
    <style>
    section[data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

set_background("assets/bckG.jpeg")

st.title("🐾 Tras las huellas del oso polar 🐻‍❄️")
st.markdown("Bienvenidos a una aventura helada en datos y machine learning.")

# Carga de datos procesados para clustering
@st.cache_data
def cargar_datos_feature():
    return pd.read_parquet("./data/processed/feature_dataset.parquet")

df = cargar_datos_feature()

# Navegación
seccion = st.sidebar.radio("Explora por sección", ["Distancia recorrida", "Patrones de movimiento","Segmentación por movimiento", "Segmentación por posGeográfica"])

if seccion == "Segmentación por movimiento":
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

        df_cluster["cluster_name"] = df_cluster["cluster"].map(nombres_ordenados)
        resumen["cluster_name"] = resumen.index.map(nombres_ordenados)

        # Visualización PCA
        # st.subheader("Visualización PCA")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
        sns.scatterplot(data=df_cluster, x="pca_1", y="pca_2", hue="cluster_name", palette="Set2", ax=ax)
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
            ax_radar.plot(angles, values, label=row["cluster_name"])
            ax_radar.fill(angles, values, alpha=0.1)

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

with st.sidebar:
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("### ❄️ Relájate con sonidos del Ártico")
    audio_file = open('assets/arctic-howling-winds-snow-ilmlcik5.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

st.sidebar.markdown("Hecho con 🐽‍❄️ por un equipo con frío pero con ganas")
