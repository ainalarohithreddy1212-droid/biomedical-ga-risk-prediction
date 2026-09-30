import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of synthetic patients
N = 500

# -----------------------------------------
# Generate synthetic biomedical features
# -----------------------------------------

age = np.random.randint(18, 81, N)

blood_pressure = np.random.normal(125, 20, N)
blood_pressure = np.clip(blood_pressure, 90, 190)

heart_rate = np.random.normal(78, 15, N)
heart_rate = np.clip(heart_rate, 50, 130)

oxygen_level = np.random.normal(97, 2.5, N)
oxygen_level = np.clip(oxygen_level, 85, 100)

temperature = np.random.normal(36.8, 0.7, N)
temperature = np.clip(temperature, 35.0, 40.0)

glucose_level = np.random.normal(110, 35, N)
glucose_level = np.clip(glucose_level, 60, 250)


# -----------------------------------------
# Create a risk score
# -----------------------------------------

risk_score = (
    0.025 * (age - 40)
    + 0.035 * (blood_pressure - 120)
    + 0.025 * (heart_rate - 75)
    + 0.35 * (95 - oxygen_level)
    + 1.5 * (temperature - 37)
    + 0.015 * (glucose_level - 100)
)


# Add random variation
risk_score += np.random.normal(0, 1.2, N)


# -----------------------------------------
# Convert score into risk class
# -----------------------------------------

risk = (risk_score > 1.5).astype(int)


# -----------------------------------------
# Create DataFrame
# -----------------------------------------

data = pd.DataFrame({
    "age": age.astype(int),
    "blood_pressure": np.round(blood_pressure, 1),
    "heart_rate": np.round(heart_rate, 1),
    "oxygen_level": np.round(oxygen_level, 1),
    "temperature": np.round(temperature, 2),
    "glucose_level": np.round(glucose_level, 1),
    "risk": risk
})


# -----------------------------------------
# Save dataset
# -----------------------------------------

data.to_csv(
    "data/patient_data.csv",
    index=False
)

print("Synthetic biomedical dataset generated successfully!")
print("Number of patients:", len(data))

print("\nRisk distribution:")
print(data["risk"].value_counts())

print("\nFirst five records:")
print(data.head())
