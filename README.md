# Space Mission Risk Analysis

A machine learning project that analyzes space mission parameters to estimate **mission risk** and explain the **key contributing factors**. The focus is on **clarity, reproducibility, and interpretability**, making this suitable for academic review and professional portfolios.

---

## Overview

Space missions involve complex interactions among technical, environmental, and operational variables. This project builds an ML pipeline to:

* Predict mission risk levels
* Identify influential factors driving risk
* Provide interpretable insights (not just predictions)

---

## Key Capabilities

* End-to-end ML workflow (data → model → insights)
* Feature importance and explainability
* Modular project structure for easy extension
* Reproducible setup with dependency management

---

## Technology Stack

* **Language:** Python
* **Core Libraries:** NumPy, Pandas, Scikit-learn
* **Explainability:** SHAP
* **Visualization:** Matplotlib / Seaborn
* **Tools:** Git, GitHub, VS Code

---

## Project Structure

```
space-mission-risk-analysis/
│── app.py                # Application entry point
│── requirements.txt      # Project dependencies
│── data/                 # Input datasets
│── models/               # Trained ML models
│── reports/              # Analysis results and plots
│── logs/                 # Execution logs
│── src/                  # Source modules
└── README.md             # Documentation
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ananyams24/space-mission-risk-analysis.git
cd space-mission-risk-analysis
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python app.py
```

The script will:

* Load and preprocess mission data
* Run the trained ML model
* Output risk predictions
* Generate explainability insights

---

## Output

* Predicted mission risk level
* Ranked list of contributing factors
* Visual explanations (SHAP plots)

---

## Project Maturity

* **Level:** Intermediate Machine Learning Project
* **Use case:** Portfolio, academic review, applied ML demonstration
* **Strengths:** Interpretability, structure, reproducibility

---

## Roadmap

* Web interface (Streamlit / FastAPI)
* Model evaluation metrics and benchmarking
* CI/CD pipeline
* Cloud deployment

---

## Author

**Ananyams24**

---

If this project is helpful, consider starring the repository.
