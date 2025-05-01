import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Charger les modèles
model = load_model("modele_lstm_eto.keras")
scaler = joblib.load("scaler_eto.pkl")

st.title("🧠 Prédiction des Besoins en Irrigation")

# Interface utilisateur
temperature = st.number_input("Température (°C)")
humidity = st.number_input("Humidité (%)")
wind_speed = st.number_input("Vitesse du vent (m/s)")
radiation = st.number_input("Rayonnement (W/m²)")

if st.button("Prédire"):
    try:
        # Création DataFrame
        input_data = pd.DataFrame([{
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "radiation": radiation
        }])

        # Normaliser
        scaled = scaler.transform(input_data)

        # Reshape pour LSTM
        X = np.reshape(scaled, (1, scaled.shape[0], scaled.shape[1]))

        # Prédiction
        prediction = model.predict(X)
        st.success(f"✅ Besoin en irrigation estimé : {prediction[0][0]:.2f}")

    except Exception as e:
        st.error(f"Erreur : {e}")
