import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ================== LOAD DATA ==================
df = pd.read_csv("dataset.csv")

# ================== FEATURE ENGINEERING ==================

# Encode operation type (Landing = 1, Takeoff = 0)
df["is_landing"] = (df["operation_type"] == "Landing").astype(int)

# Non-linear distance feature
df["distance_sq"] = df["total_distance"] ** 2

# ================== SELECT FEATURES ==================
FEATURE_COLS = [
    "total_distance",
    "distance_sq",
    "num_segments",
    "avg_length",
    "max_length",
    "turn_count",
    "is_landing"
]

X = df[FEATURE_COLS]
y = df["taxi_time"]

# ================== TRAIN / TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================== MODEL ==================
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

# ================== TRAIN ==================
model.fit(X_train, y_train)

# ================== EVALUATE ==================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f} seconds")

# ================== SAVE MODEL ==================
joblib.dump(model, "taxi_time_model.pkl")
print("✅ taxi_time_model.pkl saved")
