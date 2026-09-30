import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/patient_data.csv")

# -----------------------------
# Risk Distribution
# -----------------------------

risk_counts = data["risk"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Low Risk", "High Risk"],
    risk_counts.values
)

plt.xlabel("Risk Category")
plt.ylabel("Number of Patients")
plt.title("Biomedical Risk Distribution")

plt.tight_layout()
plt.show()


# -----------------------------
# Age vs Glucose Level
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["age"],
    data["glucose_level"],
    c=data["risk"],
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Glucose Level")
plt.title("Age vs Glucose Level by Risk")

plt.tight_layout()
plt.show()


# -----------------------------
# Blood Pressure vs Oxygen
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["blood_pressure"],
    data["oxygen_level"],
    c=data["risk"],
    alpha=0.7
)

plt.xlabel("Blood Pressure")
plt.ylabel("Oxygen Level")
plt.title("Blood Pressure vs Oxygen Level")

plt.tight_layout()
plt.show()

print("Visualizations generated successfully!")
