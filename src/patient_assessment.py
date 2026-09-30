import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data/patient_data.csv")

print("=" * 50)
print("BIOMEDICAL PATIENT RISK ASSESSMENT")
print("=" * 50)

# Example patient
patient = {
    "age": 60,
    "blood_pressure": 155,
    "heart_rate": 100,
    "oxygen_level": 92,
    "temperature": 38.0,
    "glucose_level": 175
}

print("\nPatient Information:")

for key, value in patient.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

# Simple risk scoring for demonstration
risk_score = 0

if patient["blood_pressure"] > 140:
    risk_score += 1

if patient["oxygen_level"] < 95:
    risk_score += 1

if patient["temperature"] > 37.5:
    risk_score += 1

if patient["glucose_level"] > 140:
    risk_score += 1

if patient["heart_rate"] > 100:
    risk_score += 1

print("\n" + "=" * 50)

if risk_score >= 3:
    risk_level = "HIGH"
elif risk_score >= 2:
    risk_level = "MODERATE"
else:
    risk_level = "LOW"

print("Risk Score:", risk_score, "/ 5")
print("Risk Level:", risk_level)

print("=" * 50)
print("Note: This is a synthetic research prototype,")
print("not a clinical diagnostic system.")
