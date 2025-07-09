# utils/regresion.py  
"""
Módulo Streamlit para predecir **distance_per_day_real** con el modelo
Random Forest entrenado con seis variables numéricas:

1. velocity_kmh
2. bearing_change
3. acceleration_kmph2
4. daylight_hours
5. AvgLatitude
6. AvgLongitude

Coloca los archivos:
    models/rf_daily_min.pkl         → modelo entrenado
    models/scale_daily_min.pkl      → scaler (si se usó StandardScaler)

En *app.py*:
    from utils import regresion
    ...
    elif seccion == "Distancia recorrida":
        regresion.mostrar()
"""

import streamlit as st
import pandas as pd
import pickle

# Carga del modelo + scaler (opcional)
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    with open("./models/best_reg_daily.pkl", "rb") as f:
        modelo = pickle.load(f)
    try:
        with open("./models/scale_reg_daily.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    return modelo, scaler


# Interfaz principal
def mostrar(df):
    st.markdown("## Predicción de distancia real diaria (km)")
    modelo, scaler = cargar_modelo()

    # Inputs divididos en columnas para mejor UX
    col1, col2, col3 = st.columns(3)

    with col1:
        vel   = st.number_input("Velocidad media (km/h)",   min_value=0.0,  max_value=15.0, value=2.0,  step=0.1)
        accel = st.number_input("Aceleración media (km/h²)",min_value=-5.0, max_value= 5.0, value=0.1, step=0.1)
    with col2:
        bearing = st.number_input("Cambio de dirección medio (°)", min_value=0.0, max_value=180.0, value=20.0, step=1.0)
        daylight = st.number_input("Horas de luz",            min_value=0.0, max_value=24.0, value=12.0, step=0.5)
    with col3:
        lat = st.number_input("Latitud media",  min_value=60.0, max_value=85.0,  value=72.0,  step=0.1)
        lon = st.number_input("Longitud media", min_value=-180.0, max_value=-100.0,value=-150.0, step=0.1)
    
    st.markdown("#### Estación del año")
    col_v, col_t = st.columns(2)
    with col_v:
        verano = st.checkbox("Verano", value=False)
    with col_t:
        transicion = st.checkbox("Transición", value=False)
    # Solo uno puede estar activo; si ambos, prioriza verano
    if verano:
        season_verano = 1
        season_transicion = 0
    elif transicion:
        season_verano = 0
        season_transicion = 1
    else:
        season_verano = 0
        season_transicion = 0  # es invierno

    # Orden exacto de entrenamiento
    X_new = pd.DataFrame([{ 
        "velocity_kmh":       vel,
        "bearing_change":     bearing,
        "acceleration_kmph2": accel,
        "daylight_hours":     daylight,
        "AvgLatitude":        lat,
        "AvgLongitude":       lon,
        "season_transicion":  season_transicion,
        "season_verano":      season_verano
    }])

    # Escalado si se usó en entrenamiento
    X_infer = scaler.transform(X_new) if scaler is not None else X_new

    if st.button("Predecir distancia diaria"):
        pred = float(modelo.predict(X_infer)[0])
        st.success(f"Distancia diaria estimada: **{pred:.2f} km**")


        # Título
    st.markdown("### Datos simulados de comportamiento diario del oso polar")

    # Datos
    datos = pd.DataFrame({
        "Estación": ["Verano", "Transición", "Invierno", "Verano extremo", "Estático", "Migración costera"],
        "Velocidad (km/h)": [3.5, 2.0, 0.5, 5.0, 0.0, 2.8],
        "Aceleración (km/h²)": [0.15, 0.10, 0.02, 0.30, 0.00, 0.12],
        "Cambio de dirección (°)": [35, 20, 5, 50, 0, 25],
        "Horas de luz": [20, 12, 4, 22, 8, 14],
        "Latitud": [78, 75, 73, 79, 74, 68],
        "Longitud": [-160, -155, -145, -165, -150, -135],
        "Distancia estimada (km/día)": ["60–70", "30–40", "5–8", "80–90", "~0", "45–50"],
        "Ubicación aproximada": [
            "Mar de Beaufort, Alaska",
            "Bahía de Hudson occidental",
            "Interior del Ártico canadiense",
            "Plataforma de hielo al norte de Alaska",
            "Madriguera o descanso",
            "Costa del mar de Chukotka"
        ]
    })

    # Mostrar tabla
    st.dataframe(datos)

    # # Mostrar 3 ejemplos reales al final
    # st.markdown("### Ejemplos reales (predicción vs. realidad)")
    # # Cargar datos reales
    # df = pd.read_parquet("./data/processed/feature_dataset.parquet")
    # features = [
    #     "velocity_kmh", "bearing_change", "acceleration_kmph2",
    #     "daylight_hours", "AvgLatitude", "AvgLongitude", "season_transicion", "season_verano"
    # ]

    # cat_features = ["season"]           # SIN movement_pattern

    # # Dummies para season
    # df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    # expected_columns = features + ["distance_per_day_real"]
    # missing_cols = [col for col in expected_columns if col not in df.columns]

    # if missing_cols:
    #     raise ValueError(f"Faltan columnas en el DataFrame: {missing_cols}")

    # df = df.dropna(subset=features + ["distance_per_day_real"]).sample(3, random_state=10)
    # X_ex = df[features]
    # y_ex = df["distance_per_day_real"]

    # # Escalar si procede
    # X_ex_infer = scaler.transform(X_ex) if scaler is not None else X_ex
    # y_pred = modelo.predict(X_ex_infer)

    # # Mostrar como tabla
    # st.dataframe(pd.DataFrame({
    #     "Real (km)": y_ex.values,
    #     "Predicho (km)": y_pred
    # }, index=df.index))

