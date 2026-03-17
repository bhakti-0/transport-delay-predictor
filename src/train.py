import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from src.preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/public_transport_delays.csv")

print(df.columns)

# Preprocess
df = preprocess_data(df)

# Target
y = df['actual_arrival_delay_min']  # make sure your dataset has this column

# Drop unnecessary columns
X = df.drop(['actual_arrival_delay_min', 'datetime'], axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = XGBRegressor()
model.fit(X_train, y_train)

import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.savefig("feature_importance.png")

# Save model
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")  # ✅ ADD THIS

print("Model trained and saved!")