import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ================= LOAD TRAINED MODEL =================
model = joblib.load("taxi_time_model.pkl")

# ================= LOAD DATASET (FOR MAE CHECK) =================
df = pd.read_csv("dataset.csv")

FEATURE_COLS = [
    "total_distance",
    "num_segments",
    "avg_length",
    "max_length",
    "turn_count"
]

X = df[FEATURE_COLS]
y = df["taxi_time"]

# SAME split as training (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================= MAE ON TEST DATA =================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f} seconds")

# ================= LOAD TAXIWAY LENGTHS =================
taxiway_length = pd.read_csv(
    "taxiway_lengths.csv",
    index_col=0,
    header=None
)[1].to_dict()

# ================= FEATURE EXTRACTION =================
def extract_features_from_path(path):
    lengths = [taxiway_length[p] for p in path if p in taxiway_length]

    if len(lengths) == 0:
        raise ValueError("❌ Invalid taxi path: taxiways not found")

    return pd.DataFrame([{
        "total_distance": sum(lengths),
        "num_segments": len(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "turn_count": len(lengths) - 1
    }])


# ================= USER INPUT =================
path_input = input("\nEnter taxi path (example: R-L-G): ").strip()
path = path_input.split("-")

# ================= PREDICT =================
features = extract_features_from_path(path)
predicted_time = model.predict(features)[0]

# ================= OUTPUT =================
print("\n🚕 TAXI PATH & TIME ESTIMATION")
print("Taxi Path              :", " → ".join(path))
print(f"⏱️ Estimated Taxi Time  : {predicted_time:.2f} seconds")
print(f"📊 Model MAE (reference): {mae:.2f} seconds")
