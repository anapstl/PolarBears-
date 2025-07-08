import base64
import streamlit as st
import pandas as pd
from utils import regresion, clasificacion
from utils import clustering, geo_clustering
from PIL import Image

# Configuración inicial (debe estar primero)
st.set_page_config(page_title="Exploradores Polares", layout="wide", page_icon='🐻')

# Estilo y fondo
st.markdown("""
    <style>
    section[data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# Función para establecer fondo
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
    .stMainBlockContainer  {{
        padding-top: 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Establecer fondo
set_background("assets/bckG.jpeg")

# Función de portada
def portada():
    # Carga la imagen
    # image = Image.open("./assets/mom_cubs2_adj_reduced.jpg")  # Asegúrate de que exista en esa ruta

# import streamlit as st

# Título sin markdown
    import streamlit as st
    from PIL import Image

    # Imagen local (cambia el path a tu archivo real)
    img = Image.open("assets/mom_cubs2_adj_1.jpg")
    st.image(img, width=800,  caption='TRAS LAS HUELLAS DEl OSO POLAR')

    # col1, col2 = st.columns([1, 2])

    # with col1:

    # with col2:
    #     st.markdown("""
    #         <h1 style='font-size: 48px; font-weight: 900; line-height: 1.2;'>
    #             TRAS LAS<br>
    #             <span>HUELLAS DEL</span><br>
    #             OSO POLAR
    #         </h1>
    #     """, unsafe_allow_html=True)







# Carga de datos procesados para clustering
@st.cache_data
def cargar_datos_feature():
    return pd.read_parquet("./data/processed/feature_dataset.parquet")

df = cargar_datos_feature()

# Navegación
seccion = st.sidebar.radio("Explora por sección", ["Inicio", "Distancia recorrida", "Patrones de movimiento","Segmentación por movimiento", "Agrupación geográfica"])

if seccion == "Segmentación por movimiento":
    clustering.mostrar(df)
elif seccion == "Distancia recorrida":
    regresion.mostrar(df)
elif seccion == "Patrones de movimiento":
    clasificacion.mostrar(df)
elif seccion == "Agrupación geográfica":
    geo_clustering.mostrar(df)
else:
    portada()  # Mostrar la portada si no se elige ninguna sección

# Reproductor de audio en la barra lateral
with st.sidebar:
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("### Relájate con sonidos del Ártico")
    audio_file = open('assets/arctic-howling-winds-snow-ilmlcik5.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

st.sidebar.markdown("Hecho con 🐽‍❄️ por un equipo con frío pero con ganas")
