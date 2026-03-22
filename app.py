from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

app = Flask(__name__)

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# Recreate the same model architecture from Assignment 4
class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load model
model = HousingModel()
model.load_state_dict(torch.load("house_price_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Housing Price Prediction API is running.",
        "usage": "Send a POST request to /predict with 8 housing features."
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        values = [data[feature] for feature in FEATURE_NAMES]
        input_array = np.array([values], dtype=np.float32)

        scaled_input = scaler.transform(input_array)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        return jsonify({
            "prediction": round(prediction, 4),
            "feature_order_used": FEATURE_NAMES
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing input feature: {str(e)}",
            "required_features": FEATURE_NAMES
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
