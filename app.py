import streamlit as st
import pandas as pd

# ==========================================

# PAGE CONFIGURATION

# ==========================================

st.set_page_config(
page_title="Biomedical Risk Prediction",
page_icon="🧬",
layout="wide"
)

# ==========================================

# TITLE

# ==========================================

st.title("🧬 Biomedical Risk Prediction")
st.subheader("Genetic Algorithm Based Healthcare Risk Assessment")

st.write(
"A machine-learning prototype that analyzes biomedical parameters "
"and estimates patient health-risk levels."
)

st.info(
"Research prototype using synthetic data. "
"This system is not a clinical diagnostic tool."
)

# ==========================================

# SIDEBAR

# ==========================================

st.sidebar.header("Patient Parameters")

age = st.sidebar.slider(
"Age",
min_value=1,
max_value=100,
value=60
)

blood_pressure = st.sidebar.number_input(
"Blood Pressure",
min_value=50,
max_value=250,
value=120
)

heart_rate = st.sidebar.number_input(
"Heart Rate",
min_value=30,
max_value=200,
value=80
)

oxygen_level = st.sidebar.number_input(
"Oxygen Level (%)",
min_value=50.0,
max_value=100.0,
value=98.0
)

temperature = st.sidebar.number_input(
"Temperature (°C)",
min_value=30.0,
max_value=45.0,
value=36.8
)

glucose_level = st.sidebar.number_input(
"Glucose Level",
min_value=40.0,
max_value=400.0,
value=100.0
)

# ==========================================

# PATIENT DATA

# ==========================================

patient = {
"Age": age,
"Blood Pressure": blood_pressure,
"Heart Rate": heart_rate,
"Oxygen Level": oxygen_level,
"Temperature": temperature,
"Glucose Level": glucose_level
}

# ==========================================

# RISK CALCULATION

# ==========================================

risk_score = 0

if blood_pressure > 140:
risk_score += 1

if oxygen_level < 95:
risk_score += 1

if temperature > 37.5:
risk_score += 1

if glucose_level > 140:
risk_score += 1

if heart_rate > 100:
risk_score += 1

if risk_score >= 3:
risk_level = "HIGH"
elif risk_score >= 2:
risk_level = "MODERATE"
else:
risk_level = "LOW"

# ==========================================

# MAIN DASHBOARD

# ==========================================

st.header("Patient Assessment")

col1, col2, col3 = st.columns(3)

with col1:
st.metric("Risk Score", f"{risk_score}/5")

with col2:
st.metric("Risk Level", risk_level)

with col3:
if risk_level == "HIGH":
st.error("High Risk Detected")
elif risk_level == "MODERATE":
st.warning("Moderate Risk")
else:
st.success("Low Risk")

# ==========================================

# PATIENT PARAMETERS TABLE

# ==========================================

st.header("Patient Parameters")

patient_df = pd.DataFrame(
patient.items(),
columns=["Parameter", "Value"]
)

st.dataframe(
patient_df,
use_container_width=True,
hide_index=True
)

# ==========================================

# RISK INTERPRETATION

# ==========================================

st.header("Risk Interpretation")

if risk_level == "HIGH":
st.error(
"Multiple parameters are outside the predefined prototype "
"thresholds. Further evaluation may be required."
)

elif risk_level == "MODERATE":
st.warning(
"Some parameters are outside the predefined prototype "
"thresholds."
)

else:
st.success(
"The entered parameters remain within the predefined "
"prototype thresholds."
)

# ==========================================

# PROJECT INFORMATION

# ==========================================

st.header("About the Project")

st.write(
"""
This project explores the use of a Genetic Algorithm for
biomedical feature selection. The system works with synthetic
patient data containing age, blood pressure, heart rate,
oxygen level, temperature and glucose level.

```
The Genetic Algorithm searches for useful combinations of
biomedical features, while machine learning is used for
classification and risk prediction.
"""
```

)

st.caption(
"OptiForge 2026 | IEEE EMBS × IEEE CIS | Vardhaman College of Engineering"
)
