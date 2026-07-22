import os
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from schema import validate_and_extract, ValidationError

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "iris_model.pkl"
model = joblib.load(MODEL_PATH)

FLOWER_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route("/")
def home():
    return jsonify({"message": "Iris Prediction API is running!"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        features = validate_and_extract(data)
        prediction = model.predict(features)[0]
        return jsonify({"class": FLOWER_NAMES[int(prediction)]}), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)