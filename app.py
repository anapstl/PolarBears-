import base64
import streamlit as st
import pandas as pd
from utils import regresion, clasificacion, clustering

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
# st.logo("../assets/mom_cubs2_adj_reduced.ico")

st.markdown("""
    <style>
    section[data-testid="stSidebarContent"] {
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# Llama esta función al inicio de tu app
# set_background("assets/mom_cubs2_adj_1.jpg")
set_background("assets/bckG.jpeg")

# Encabezado principal
st.title("🐾 Tras las huellas del oso polar 🐻‍❄️")
st.markdown("Bienvenidos a una aventura helada en datos y machine learning.")

# Carga de datos ficticios
# @st.cache_data
def cargar_datos():
    return pd.read_csv("./data/raw/polarBear_CTCRWlocations_chukchiBeaufort_1985-2017.csv")

df = cargar_datos()

# Navegación
seccion = st.sidebar.radio("Explora por sección", ["Distancia recorrida", "Patrones de movimiento", "Segmentación por estilo"])

with st.sidebar:
    # Otros controles o menús aquí...

    # Espacio para empujar el audio hacia abajo
    st.markdown("<br>" * 8, unsafe_allow_html=True)

    st.markdown("### 🌨️ Relájate con sonidos del Ártico")

    # Cargar archivo de sonido
    audio_file = open('assets/arctic-howling-winds-snow-ilmlcik5.wav', 'rb')
    audio_bytes = audio_file.read()    

    st.audio(audio_bytes, format='audio/mp3')

# if seccion == "Distancia recorrida":
#     regresion.mostrar(df)
# elif seccion == "Patrones de movimiento":
#     clasificacion.mostrar(df)
# elif seccion == "Segmentación por estilo":
#     clustering.mostrar(df)

st.sidebar.markdown("Hecho con 🐻‍❄️ por un equipo con frío pero con ganas")
