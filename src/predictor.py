"""
Biomedical risk prediction using Genetic Algorithm feature selection.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ga_feature_selection import genetic_feature_selection


FEATURE_NAMES: List[str] = [
    "age",
    "blood_pressure",
    "heart_rate",
    "oxygen_level",
    "temperature",
    "glucose_level",
]


def train_ga_model(
    data_path: str = "data/patient_data.csv",
) -> Tuple[
    LogisticRegression,
    StandardScaler,
    np.ndarray,
    float,
    List[float],
]:
    """
    Train a Logistic Regression model using GA-selected features.

    Parameters
    ----------
    data_path : str
        Path to the biomedical patient dataset.

    Returns
    -------
    tuple
        Trained model, scaler, selected feature indices,
        best GA fitness, and fitness history.
    """

    data = pd.read_csv(data_path)

    X = data[FEATURE_NAMES].values
    y = data["risk"].values

    best_chromosome, best_fitness, history = (
        genetic_feature_selection(
            X,
            y,
            population_size=10,
            generations=10,
            mutation_rate=0.1,
        )
    )

    selected_indices = np.where(
        best_chromosome == 1
    )[0]

    # Safety fallback if GA selects no features
    if len(selected_indices) == 0:
        selected_indices = np.arange(
            X.shape[1]
        )

    X_selected = X[:, selected_indices]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_selected
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_scaled,
        y,
    )

    return (
        model,
        scaler,
        selected_indices,
        best_fitness,
        history,
    )


def predict_patient(
    model: LogisticRegression,
    scaler: StandardScaler,
    selected_indices: np.ndarray,
    patient: Dict[str, float],
) -> Tuple[int, float]:
    """
    Predict biomedical risk for a patient.

    Parameters
    ----------
    model : LogisticRegression
        Trained prediction model.

    scaler : StandardScaler
        Feature scaler used during training.

    selected_indices : np.ndarray
        Features selected by the Genetic Algorithm.

    patient : Dict[str, float]
        Patient biomedical measurements.

    Returns
    -------
    tuple
        Predicted class and high-risk probability.
    """

    patient_values = np.array(
        [[
            patient["age"],
            patient["blood_pressure"],
            patient["heart_rate"],
            patient["oxygen_level"],
            patient["temperature"],
            patient["glucose_level"],
        ]]
    )

    selected_values = patient_values[
        :, selected_indices
    ]

    scaled_values = scaler.transform(
        selected_values
    )

    prediction = int(
        model.predict(scaled_values)[0]
    )

    probability = float(
        model.predict_proba(
            scaled_values
        )[0][1]
    )

    return prediction, probability
