import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# ================== LOAD DATA ==================
df = pd.read_csv("dataset.csv")

# ================== SELECT NUMERIC FEATURES ONLY ==================
FEATURE_COLS = [
    "total_distance",
    "num_segments",
    "avg_length",
    "max_length",
    "turn_count"
]

X = df[FEATURE_COLS]
y = df["taxi_time"]

# ================== TRAIN / TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================== MODEL ==================
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ================== TRAIN ==================
model.fit(X_train, y_train)

# ================== EVALUATE ==================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Mean Absolute Error (MAE): {mae:.2f} seconds")

# ================== SAVE MODEL ==================
joblib.dump(model, "taxi_time_model.pkl")
print("✅ taxi_time_model.pkl saved")
