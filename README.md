# 🧬 Biomedical Risk Prediction Using Genetic Algorithm

## OptiForge 2026 — IEEE EMBS × IEEE CIS

A machine learning prototype that uses a **Genetic Algorithm (GA)** for selecting relevant biomedical features and a classification model for predicting patient health-risk levels.

> **Note:** This project uses a synthetic dataset and is intended for research and educational purposes. It is **not a clinical diagnostic system**.

---

## 📌 Problem Statement

Healthcare datasets can contain multiple patient parameters such as age, blood pressure, heart rate, oxygen level, temperature, and glucose level.

Using all available parameters may introduce unnecessary features and increase model complexity.

The objective of this project is to develop a computational approach that:

* Identifies relevant biomedical features.
* Uses a Genetic Algorithm for feature selection.
* Builds a machine-learning-based risk prediction model.
* Provides an interpretable risk assessment.
* Visualizes patient data and model performance.

---

## 💡 Proposed Solution

Our system combines:

**Synthetic Biomedical Dataset → Genetic Algorithm → Feature Selection → Machine Learning → Risk Prediction**

The Genetic Algorithm searches through different combinations of biomedical features and identifies a feature subset with strong predictive performance.

---

## 🧬 Genetic Algorithm

The Genetic Algorithm represents each possible feature combination as a binary chromosome.

Example:

```text
[1, 0, 1, 0, 1, 0]
```

Where:

```text
1 = Feature selected
0 = Feature not selected
```

### GA Process

1. Initialize a population of feature combinations.
2. Evaluate the fitness of each chromosome.
3. Select the better-performing solutions.
4. Perform crossover.
5. Apply mutation.
6. Generate a new population.
7. Repeat for multiple generations.
8. Select the best feature combination.

---

## 🏥 Biomedical Features

The synthetic dataset contains:

| Feature        | Description                |
| -------------- | -------------------------- |
| Age            | Patient age                |
| Blood Pressure | Blood pressure measurement |
| Heart Rate     | Heart rate                 |
| Oxygen Level   | Blood oxygen saturation    |
| Temperature    | Body temperature           |
| Glucose Level  | Blood glucose level        |
| Risk           | Target risk category       |

---

## 📊 Dataset

A synthetic dataset containing **500 patient records** was generated for this prototype.

Risk distribution:

* Low Risk: 396
* High Risk: 104

The dataset is generated programmatically and does not contain real patient information.

---

## 🤖 Machine Learning

The project currently uses **Logistic Regression** as the classification model.

The Genetic Algorithm is used to identify useful feature combinations before the final prediction stage.

---

## 📈 Current Prototype Result

The current prototype achieved approximately:

**83.2% classification accuracy**

The exact result may vary when the dataset or model configuration is changed.

Additional evaluation metrics are provided through:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📁 Project Structure

```text
biomedical-ga-risk-prediction/
│
├── data/
│   └── patient_data.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── risk_prediction.py
│   ├── visualize_results.py
│   ├── patient_assessment.py
│   └── model_evaluation.py
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Genetic Algorithm
* Logistic Regression

---

## 🚀 Features

### 1. Synthetic Dataset Generation

Automatically generates biomedical patient records for experimentation.

### 2. Genetic Feature Selection

Uses evolutionary optimization to search for useful feature combinations.

### 3. Risk Prediction

Predicts whether a patient belongs to a low-risk or high-risk category.

### 4. Patient Assessment

Provides a simple parameter-based risk assessment.

### 5. Data Visualization

Visualizes:

* Risk distribution
* Age vs glucose level
* Blood pressure vs oxygen level

### 6. Model Evaluation

Provides:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/ainalarohithreddy1212-droid/biomedical-ga-risk-prediction.git
```

Move into the project:

```bash
cd biomedical-ga-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python3 src/generate_dataset.py
```

Run the Genetic Algorithm:

```bash
python3 src/risk_prediction.py
```

Run visualizations:

```bash
python3 src/visualize_results.py
```

Run patient assessment:

```bash
python3 src/patient_assessment.py
```

Evaluate the model:

```bash
python3 src/model_evaluation.py
```

---

## 🔬 Future Scope

Future versions could include:

* Larger real-world biomedical datasets.
* Multiple machine-learning models.
* Improved Genetic Algorithm optimization.
* Explainable AI techniques.
* Real-time patient monitoring.
* IoT-based health sensors.
* Secure healthcare data storage.
* Web-based patient dashboard.
* Model comparison and hyperparameter optimization.

---

## ⚠️ Disclaimer

This project is a **research and educational prototype** developed using synthetic data.

It should not be used for medical diagnosis, treatment decisions, or emergency healthcare decisions.

---

## 👥 Team

**OptiForge 2026**

IEEE EMBS × IEEE CIS
Vardhaman College of Engineering
