import streamlit as st, pandas as pd, pickle, numpy as np

@st.cache_resource
def cargar_modelo():
    with open("./models/rf_daily.pkl","rb") as f:  model = pickle.load(f)
    with open("./models/scale_daily.pkl","rb") as f: scaler = pickle.load(f)
    return model, scaler

def mostrar(df):
    st.markdown("## Predicción de distancia real diaria")

    model, scaler = cargar_modelo()

    # ─ inputs
    col1,col2 = st.columns(2)
    with col1:
        vel   = st.number_input("Velocidad media (km/h)", 0.0, 10.0, 2.0, 0.1)
        bear  = st.number_input("Cambio dirección medio (°)", 0.0, 180.0, 20.0, 1.0)
        accel = st.number_input("Aceleración media (km/h²)", -5.0, 5.0, 0.1, 0.1)
    with col2:
        lat   = st.number_input("Latitud media", 60.0, 85.0, 72.0, 0.1)
        lon   = st.number_input("Longitud media", -180.0, -100.0, -150.0, 0.1)
        light = st.number_input("Horas de luz", 0.0, 24.0, 12.0, 0.5)

    season  = st.selectbox("Estación", ["invierno","transicion","verano"])
    pattern = st.selectbox("Patrón movimiento", ["activo","estacionario"])

    # ─ construcción X_new (orden según entrenamiento)
    X_new = pd.DataFrame([{
        "velocity_kmh": vel,
        "acceleration_kmph2": accel,
        "bearing_change": bear,
        "daylight_hours": light,
        "is_polar_night": int(light==0),
        "is_midnight_sun": int(light==24),
        "AvgLatitude": lat,
        "AvgLongitude": lon,
        # dummies
        "season_transicion": int(season=="transicion"),
        "season_verano": int(season=="verano"),
        "movement_pattern_estacionario": int(pattern=="estacionario")
    }])

    # ─ predicción
    if st.button("Predecir"):
        pred = model.predict(scaler.transform(X_new))[0]
        st.success(f"Distancia real diaria estimada: **{pred:.2f} km**")
