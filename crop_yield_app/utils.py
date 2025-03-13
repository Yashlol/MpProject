import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "data_core.csv")
df = pd.read_csv(file_path)

# Rename incorrect column name
df.rename(columns={"Temparature": "Temperature"}, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["Soil Type", "Fertilizer Name", "Crop Type"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=["Crop Type"])
y = df["Crop Type"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
model_path = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")

joblib.dump(model, model_path)
joblib.dump(label_encoders, encoder_path)

print("Model and encoders saved successfully!")
