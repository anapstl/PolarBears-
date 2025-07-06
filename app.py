import base64
import streamlit as st
import pandas as pd
from utils import regresion, clasificacion
from utils import clustering

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
    clustering.mostrar(df)
elif seccion == "Distancia recorrida":
    regresion.mostrar(df)

with st.sidebar:
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("### ❄️ Relájate con sonidos del Ártico")
    audio_file = open('assets/arctic-howling-winds-snow-ilmlcik5.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

st.sidebar.markdown("Hecho con 🐽‍❄️ por un equipo con frío pero con ganas")
