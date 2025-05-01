from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle
model = load_model("modele_lstm_eto.keras")

# Charger le scaler
scaler = joblib.load("scaler_eto.pkl")  # doit être généré lors de l'entraînement

# Endpoint de test
@app.route("/")
def home():
    return "API de prédiction des besoins en irrigation"

# Endpoint de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Convertir en DataFrame
        df = pd.DataFrame([data])

        # Normaliser
        scaled = scaler.transform(df)

        # Reformater pour LSTM [1, timesteps, features]
        X = np.reshape(scaled, (1, scaled.shape[0], scaled.shape[1]))

        # Prédire
        prediction = model.predict(X)
        return jsonify({"prediction": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# Lancer l'app
if __name__ == "__main__":
    app.run(debug=True)