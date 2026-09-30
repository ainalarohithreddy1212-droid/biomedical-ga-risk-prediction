import numpy as np
import pandas as pd
import streamlit as st

from src.predictor import (
    FEATURE_NAMES,
    predict_patient,
    train_ga_model,
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Biomedical Risk Prediction",
    page_icon="🧬",
    layout="wide",
)


# ==========================================
# TITLE
# ==========================================

st.title("🧬 Biomedical Risk Prediction")

st.subheader(
    "Genetic Algorithm Based Feature Selection "
    "for Biomedical Risk Assessment"
)

st.info(
    "This is a synthetic research prototype and "
    "is not a clinical diagnostic system."
)


# ==========================================
# TRAIN GA + ML MODEL
# ==========================================

@st.cache_resource
def load_model():
    """Train and cache the GA-based prediction model."""

    return train_ga_model(
        "data/patient_data.csv"
    )


model, scaler, selected_indices, best_fitness, history = (
    load_model()
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=60,
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120,
)

heart_rate = st.sidebar.number_input(
    "Heart Rate",
    min_value=30,
    max_value=220,
    value=80,
)

oxygen_level = st.sidebar.number_input(
    "Oxygen Level (%)",
    min_value=50.0,
    max_value=100.0,
    value=98.0,
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=30.0,
    max_value=45.0,
    value=37.0,
)

glucose_level = st.sidebar.number_input(
    "Glucose Level",
    min_value=40,
    max_value=500,
    value=100,
)


# ==========================================
# PATIENT DATA
# ==========================================

patient = {
    "age": float(age),
    "blood_pressure": float(blood_pressure),
    "heart_rate": float(heart_rate),
    "oxygen_level": float(oxygen_level),
    "temperature": float(temperature),
    "glucose_level": float(glucose_level),
}


# ==========================================
# PREDICTION
# ==========================================

prediction, probability = predict_patient(
    model,
    scaler,
    selected_indices,
    patient,
)


# ==========================================
# RISK CLASSIFICATION
# ==========================================

if probability >= 0.66:
    risk_level = "HIGH"
elif probability >= 0.33:
    risk_level = "MODERATE"
else:
    risk_level = "LOW"


# ==========================================
# DASHBOARD
# ==========================================

st.header("Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Risk Level",
        risk_level,
    )

with col2:
    st.metric(
        "Risk Probability",
        f"{probability * 100:.2f}%",
    )

with col3:
    st.metric(
        "GA Fitness",
        f"{best_fitness * 100:.2f}%",
    )


# ==========================================
# PATIENT PARAMETERS
# ==========================================

st.header("Patient Parameters")

patient_table = pd.DataFrame(
    {
        "Parameter": [
            "Age",
            "Blood Pressure",
            "Heart Rate",
            "Oxygen Level",
            "Temperature",
            "Glucose Level",
        ],
        "Value": [
            age,
            blood_pressure,
            heart_rate,
            oxygen_level,
            temperature,
            glucose_level,
        ],
    }
)

st.dataframe(
    patient_table,
    use_container_width=True,
    hide_index=True,
)


# ==========================================
# GA SELECTED FEATURES
# ==========================================

st.header("Genetic Algorithm Feature Selection")

selected_features = [
    FEATURE_NAMES[index]
    for index in selected_indices
]

st.write(
    "The Genetic Algorithm selected the following "
    "features for prediction:"
)

st.success(
    ", ".join(selected_features)
)


# ==========================================
# GA FITNESS PROGRESS
# ==========================================

st.header("GA Optimization Progress")

fitness_data = pd.DataFrame(
    {
        "Generation": np.arange(
            1,
            len(history) + 1
        ),
        "Best Fitness": history,
    }
)

st.line_chart(
    fitness_data.set_index("Generation")
)


# ==========================================
# INTERPRETATION
# ==========================================

st.header("Risk Interpretation")

if risk_level == "HIGH":
    st.error(
        "The model estimates a relatively high "
        "risk probability for the entered synthetic "
        "patient profile."
    )

elif risk_level == "MODERATE":
    st.warning(
        "The model estimates a moderate "
        "risk probability for the entered "
        "synthetic patient profile."
    )

else:
    st.success(
        "The model estimates a relatively low "
        "risk probability for the entered "
        "synthetic patient profile."
    )


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.header("About the Project")

st.write(
    """
    This project demonstrates a Genetic Algorithm based
    feature-selection approach for biomedical risk prediction.

    The Genetic Algorithm searches through combinations of
    biomedical features and identifies a useful subset.

    The selected features are then provided to a
    Logistic Regression model for risk prediction.

    The current dataset contains synthetic patient records
    with age, blood pressure, heart rate, oxygen level,
    temperature, and glucose level.
    """
)


# ==========================================
# DISCLAIMER
# ==========================================

st.caption(
    "OptiForge 2026 | IEEE EMBS × IEEE CIS | "
    "Vardhaman College of Engineering"
)

st.caption(
    "Research prototype using synthetic data. "
    "Not intended for medical diagnosis or clinical decisions."
)
