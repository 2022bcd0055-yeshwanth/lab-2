import pandas as pd
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Base paths (CI-safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality-red.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure outputs directory exists (CRITICAL for GitHub)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH, sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (required for job summary)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))

# Save metrics
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)