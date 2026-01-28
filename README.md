# 🚀 Space Mission Risk Analysis

## Project at a Glance (60-Second Overview)

This project is an **end-to-end machine learning system** that predicts the **operational risk of space missions** before launch using historical mission data and mission parameters.

**Input:** Mission parameters (CSV / API / Dashboard)

* Payload mass
* Fuel level
* Launch vehicle
* Weather conditions
* Mission type & history

**Output:**

* Risk score (0–1)
* Risk category: **Low / Medium / High**
* Explainability using SHAP (why a mission is risky)

**Model:** Random Forest Classifier
**Best ROC-AUC:** ~0.91
**Use case:** Decision support for mission planners to identify high-risk missions early

---

## Why This Project Matters

Space missions are **high-cost and high-risk**. Even small misjudgments in payload, fuel, or environmental conditions can lead to mission failure.

This system helps:

* Detect risky missions **before launch**
* Support data-driven decision making
* Improve mission safety and planning

This is not just an ML demo — it is a **risk assessment decision-support system**.

---

## System Architecture

```
        Raw Mission Data (CSV)
                │
                ▼
        Data Preprocessing
        (Cleaning, Encoding)
                │
                ▼
        Feature Engineering
                │
                ▼
        ML Model Training
        (Random Forest)
                │
                ▼
        Risk Prediction
        (Score + Category)
                │
                ▼
        SHAP Explainability
        (Feature Importance)
                │
                ▼
        Streamlit Dashboard / API
```

---

## Tech Stack

* **Python**
* **Pandas, NumPy** – Data processing
* **Scikit-learn** – Machine learning
* **SHAP** – Model explainability
* **Matplotlib / Seaborn** – Visualization
* **Streamlit** – Interactive dashboard

---

## Model Performance

| Metric              | Value                          |
| ------------------- | ------------------------------ |
| Accuracy            | ~87%                           |
| ROC-AUC             | ~0.91                          |
| False Negative Rate | Low (priority for risk models) |

> Evaluation focuses on minimizing missed high-risk missions.

---

## How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python src/train.py
```

### 3. Run the dashboard

```
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
space-mission-risk-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── models/
├── app/
│   └── streamlit_app.py
├── tests/
├── requirements.txt
└── README.md
```

---

## Explainability & Trust

This project uses **SHAP values** to explain:

* Which features contribute most to risk
* Why a specific mission is classified as high-risk

Explainability is critical for **high-stakes domains like aerospace**.

---

## Limitations & Future Work

**Current limitations:**

* Synthetic / limited historical data
* No real-time telemetry integration

**Future improvements:**

* Real mission datasets
* FastAPI deployment
* CI/CD with automated testing
* Model monitoring & drift detection

---

## Ideal Use Cases

* Aerospace analytics demos
* Risk assessment ML systems
* Final-year / capstone project
* Entry-level ML engineer or data scientist portfolio

---

## Author

**Ananya M S**
Machine Learning & Data Science Enthusiast

---

⭐ If you are a recruiter: this project demonstrates **end-to-end ML development, explainability, and deployment readiness**.
