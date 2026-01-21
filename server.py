from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # 🔥 VERY IMPORTANT for browser requests

# ================= PATH SAFETY =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "taxi_time_model.pkl")
TAXIWAY_PATH = os.path.join(BASE_DIR, "taxiway_lengths.csv")

# ================= LOAD MODEL =================
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ================= LOAD TAXIWAY LENGTHS =================
try:
    taxiway_length = pd.read_csv(
        TAXIWAY_PATH,
        index_col=0,
        header=None
    )[1].to_dict()
except Exception as e:
    raise RuntimeError(f"❌ Failed to load taxiway lengths: {e}")

# ================= FEATURE EXTRACTION =================
def extract_features_from_path(path):
    lengths = []

    for p in path:
        if p not in taxiway_length:
            return None
        lengths.append(taxiway_length[p])

    if len(lengths) == 0:
        return None

    return pd.DataFrame([{
        "total_distance": sum(lengths),
        "num_segments": len(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "turn_count": len(lengths) - 1
    }])

# ================= API =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "path" not in data:
            return jsonify({"error": "Taxi path not provided"}), 400

        path_input = data["path"].strip()
        path = path_input.split("-")

        features = extract_features_from_path(path)
        if features is None:
            return jsonify({"error": "Invalid taxi path"}), 400

        prediction = model.predict(features)[0]

        return jsonify({
            "taxi_path": " → ".join(path),
            "predicted_time": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
