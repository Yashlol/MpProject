import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv("C:\\Users\\Yash\\OneDrive\\Desktop\\Django\\MpProject\\crop_yield_app\\data_core.csv")  # Update the correct path

# Select input features (excluding soil type)
features = ["temperature", "humidity", "moisture", "nitrogen", "phosphorus", "potassium"]
X = data[features]

# Target variable: Soil type
y = data["Soil_Type"]  # Ensure this column exists in your dataset

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "soil_model.pkl")
print("Model training complete. Saved as soil_model.pkl")
