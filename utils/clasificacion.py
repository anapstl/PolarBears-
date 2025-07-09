# utils/clasificacion.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource(show_spinner=False)
def cargar_modelo_y_scaler():
    try:
        with open("./models/cls_pattern.pkl", "rb") as f:
            modelo = pickle.load(f)
    except Exception:
        modelo = None
    try:
        with open("./models/cls_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    return modelo, scaler

def mostrar(df):
    st.markdown("## Clasificación de patrón de movimiento (activo/estacionario)")

    modelo, scaler = cargar_modelo_y_scaler()

    col1, col2, col3 = st.columns(3)
    with col1:
        vel = st.number_input("Velocidad media (km/h)", 0.0, 15.0, 2.0, 0.1)
        accel = st.number_input("Aceleración media (km/h²)", -5.0, 5.0, 0.1, 0.1)
    with col2:
        bearing = st.number_input("Cambio dirección medio (°)", 0.0, 180.0, 20.0, 1.0)
        daylight = st.number_input("Horas de luz", 0.0, 24.0, 12.0, 0.5)
    with col3:
        lat = st.number_input("Latitud media", 60.0, 85.0, 72.0, 0.1)
        lon = st.number_input("Longitud media", -180.0, -100.0, -150.0, 0.1)

    st.markdown("#### Estación del año")
    col_v, col_t = st.columns(2)
    with col_v:
        verano = st.checkbox("Verano", value=False)
    with col_t:
        transicion = st.checkbox("Transición", value=False)
    if verano:
        season_verano = 1
        season_transicion = 0
    elif transicion:
        season_verano = 0
        season_transicion = 1
    else:
        season_verano = 0
        season_transicion = 0

    X_new = pd.DataFrame([{ 
        "velocity_kmh": vel,
        "bearing_change": bearing,
        "acceleration_kmph2": accel,
        "daylight_hours": daylight,
        "AvgLatitude": lat,
        "AvgLongitude": lon,
        "season_transicion": season_transicion,
        "season_verano": season_verano
    }])

    if scaler is not None:
        X_infer = scaler.transform(X_new)
    else:
        X_infer = X_new

    if st.button("Predecir patrón de movimiento"):
        if modelo is not None:
            prob = modelo.predict_proba(X_infer)[0][1]
            pred = modelo.predict(X_infer)[0]
            label = "Estacionario" if pred == 1 else "Activo"
            st.success(f"Predicción: **{label}** (probabilidad de 'estacionario': {prob:.2f})")
        else:
            st.error("Modelo no encontrado. Entrena y guarda primero el modelo en 'models/logreg_movement.pkl'.")


    # Título
    st.markdown("## Datos simulados para clasificación de actividad de osos polares")

    # Datos simulados
    datos_clasificacion = pd.DataFrame({
        "Estación": ["Verano", "Transición", "Invierno", "Verano extremo", "Estático", "Migración costera"],
        "Velocidad (km/h)": [3.5, 2.0, 0.5, 5.0, 0.0, 2.8],
        "Aceleración (km/h²)": [0.15, 0.10, 0.02, 0.30, 0.00, 0.12],
        "Cambio de dirección (°)": [35, 20, 5, 50, 0, 25],
        "Horas de luz": [20, 12, 4, 22, 8, 14],
        "Latitud": [78, 75, 73, 79, 74, 68],
        "Longitud": [-160, -155, -145, -165, -150, -135],
        "Patrón de movimiento": [
            "Activo", "Activo", "Estacionario", "Activo", "Estacionario", "Activo"
        ],
        "Ubicación aproximada": [
            "Mar de Beaufort, Alaska",
            "Bahía de Hudson occidental",
            "Interior del Ártico canadiense",
            "Plataforma de hielo al norte de Alaska",
            "Madriguera o descanso",
            "Costa del mar de Chukotka"
        ]
    })

    # Mostrar tabla en app
    st.dataframe(datos_clasificacion)


    # # Mostrar ejemplos reales
    # st.markdown("### Ejemplos reales (predicción vs. realidad)")
    # try:
    #     df = pd.read_parquet("data/processed/feature_dataset.parquet")
    #     df["movement_pattern_bin"] = (df["movement_pattern"] == "estacionario").astype(int)
    #     features = [
    #         "velocity_kmh", "bearing_change", "acceleration_kmph2",
    #         "daylight_hours", "AvgLatitude", "AvgLongitude",
    #         "season_verano", "season_transicion"
    #     ]
    #     df_ex = df.dropna(subset=features + ["movement_pattern_bin"]).sample(3, random_state=None)
    #     X_ex = df_ex[features]
    #     y_ex = df_ex["movement_pattern_bin"]
    #     X_ex_infer = scaler.transform(X_ex) if scaler is not None else X_ex
    #     y_pred = modelo.predict(X_ex_infer)
    #     df_show = pd.DataFrame({
    #         "Real": y_ex.map({1: "Estacionario", 0: "Activo"}).values,
    #         "Predicho": pd.Series(y_pred).map({1: "Estacionario", 0: "Activo"}).values
    #     }, index=df_ex.index)
    #     st.dataframe(df_show)
    # except Exception as e:
    #     st.info(f"No se pueden mostrar ejemplos: {e}")
