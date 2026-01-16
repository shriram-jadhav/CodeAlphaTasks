import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Load dataset
data = pd.read_csv("data/car_data.csv")

# Drop unnecessary column
data.drop('Car_Name', axis=1, inplace=True)

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Features & target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model safely
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/car_price_model.joblib")

print("Model saved using joblib")
