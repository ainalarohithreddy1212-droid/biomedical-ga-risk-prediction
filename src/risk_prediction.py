import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------
# 1. Load the biomedical dataset
# ---------------------------------------

data = pd.read_csv("data/patient_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# ---------------------------------------
# 2. Separate features and target
# ---------------------------------------

X = data.drop("risk", axis=1)
y = data["risk"]

# ---------------------------------------
# 3. Split dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# 4. Normalize the data
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------
# 5. Train risk prediction model
# ---------------------------------------

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# ---------------------------------------
# 6. Evaluate model
# ---------------------------------------

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ---------------------------------------
# 7. Predict risk for a new patient
# ---------------------------------------

new_patient = np.array([
    [60, 155, 100, 92, 38.0, 175]
])

new_patient_scaled = scaler.transform(new_patient)

risk_prediction = model.predict(new_patient_scaled)[0]
risk_probability = model.predict_proba(new_patient_scaled)[0][1]

print("\n--------------------------------")
print("NEW PATIENT RISK ASSESSMENT")
print("--------------------------------")

if risk_prediction == 1:
    print("Risk Level: HIGH")
else:
    print("Risk Level: LOW")

print(
    "Estimated risk probability:",
    round(risk_probability * 100, 2),
    "%"
)
