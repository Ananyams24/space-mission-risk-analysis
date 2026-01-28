🚀 Space Mission Risk Analysis
Project at a Glance

An end-to-end machine learning system that predicts the operational risk of space missions before launch using mission parameters and historical data.

Input: Mission parameters (CSV / Streamlit Dashboard)

Payload mass

Fuel level

Launch vehicle

Weather conditions

Mission type & history

Output:

Risk score (0–1)

Risk category: Low / Medium / High

SHAP-based explainability

Model: Random Forest Classifier
Test ROC-AUC: ~0.87
Accuracy: ~0.90
Use case: Decision support for identifying high-risk missions early

Why It Matters

Space missions are high-cost and high-risk. This system helps mission planners:

Detect risky missions before launch

Make data-driven decisions

Improve mission safety and planning

This is a risk assessment decision-support system, not just an ML demo.

System Architecture
Raw Data → Preprocessing → Feature Engineering
        → ML Model (Random Forest)
        → Risk Prediction
        → SHAP Explainability
        → Streamlit Dashboard

Tech Stack

Python, Pandas, NumPy

Scikit-learn

SHAP

Matplotlib

Streamlit

Model Transparency

Model version: v1.0.0

Training records: ~4,198 missions

Test ROC-AUC tracked in dashboard

Classification report & confusion matrix available

How to Run
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py

Limitations & Future Work

Synthetic / limited historical data

No real-time telemetry

Planned:

FastAPI deployment

CI/CD & automated testing

Model monitoring & drift detection

Author

Ananya M S
Machine Learning & Data Science Enthusiast

⭐ This project demonstrates applied ML, model evaluation, and explainability in a high-stakes domain.
