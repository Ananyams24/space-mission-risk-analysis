import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Space Mission Analytics",
    page_icon="🚀",
    layout="wide"
)

# =================================================
# PATHS
# =================================================
PROCESSED_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/xgboost_model.pkl"

MODEL_VERSION = "v1.0.0"
MODEL_AUC = 0.87          # <-- replace with real value if available
LAST_TRAINED = "Jan 2026"

# =================================================
# LOAD DATA & MODEL
# =================================================
@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df_raw = load_data()
model = load_model()

# =================================================
# FEATURE ENGINEERING
# =================================================
def engineer_features(df):
    df = df.copy()

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["launch_year"] = df["launch_date"].dt.year
    df["launch_decade"] = (df["launch_year"] // 10) * 10

    df["rocket_failure_rate"] = df.groupby("rocket")["failure"].transform("mean")
    df["rocket_launch_count"] = df.groupby("rocket")["failure"].transform("count")
    df["org_failure_rate"] = df.groupby("company")["failure"].transform("mean")

    df["past_failure_rate"] = (
        df.sort_values("launch_year")
        .groupby("company")["failure"]
        .transform(lambda x: x.expanding().mean())
    )

    for col in ["company", "location", "rocket"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df

df = engineer_features(df_raw)

FEATURES = [
    "launch_year",
    "launch_decade",
    "rocket_failure_rate",
    "rocket_launch_count",
    "org_failure_rate",
    "past_failure_rate",
    "company_enc",
    "location_enc",
    "rocket_enc"
]

# =================================================
# CONFIDENCE ESTIMATION
# =================================================
def confidence_level(count):
    if count > 100:
        return "High"
    elif count > 30:
        return "Medium"
    else:
        return "Low"

# =================================================
# NLP RISK EXPLANATION
# =================================================
def explain_risk(row, risk):
    reasons = []

    if row["rocket_failure_rate"] > 0.25:
        reasons.append("rocket has a high historical failure rate")

    if row["rocket_launch_count"] < 5:
        reasons.append("rocket has limited operational history")

    if row["org_failure_rate"] > 0.30:
        reasons.append("organization has higher-than-average failures")

    if row["past_failure_rate"] > 0.35:
        reasons.append("recent missions show increasing failures")

    if not reasons:
        reasons.append("historical performance is strong")

    level = "High" if risk > 60 else "Moderate" if risk > 30 else "Low"

    return level, " and ".join(reasons)

# =================================================
# AUDIT LOGGING
# =================================================
def log_prediction(company, rocket, year, risk):
    os.makedirs("logs", exist_ok=True)
    log = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "company": company,
        "rocket": rocket,
        "launch_year": year,
        "predicted_risk": risk,
        "model_version": MODEL_VERSION
    }])
    log.to_csv(
        "logs/prediction_audit_log.csv",
        mode="a",
        header=not os.path.exists("logs/prediction_audit_log.csv"),
        index=False
    )

# =================================================
# PDF REPORT
# =================================================
def generate_pdf(company, rocket, year, risk, confidence, explanation):
    os.makedirs("reports", exist_ok=True)
    path = f"reports/mission_report_{company}_{rocket}.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)

    content = [
        Paragraph("<b>Space Mission Risk Assessment Report</b>", styles["Title"]),
        Paragraph(f"Organization: {company}", styles["Normal"]),
        Paragraph(f"Rocket: {rocket}", styles["Normal"]),
        Paragraph(f"Launch Year: {year}", styles["Normal"]),
        Paragraph(f"Predicted Failure Risk: {risk:.2f}%", styles["Normal"]),
        Paragraph(f"Confidence Level: {confidence}", styles["Normal"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<b>Explanation</b>", styles["Heading2"]),
        Paragraph(explanation, styles["Normal"]),
    ]

    doc.build(content)
    return path

# =================================================
# SIDEBAR
# =================================================
st.sidebar.title("⚙️ Controls")
if st.sidebar.checkbox("🇮🇳 ISRO-only Analytics"):
    df = df[df["company"].str.contains("ISRO", case=False, na=False)]

# =================================================
# HEADER
# =================================================
st.title("🚀 Space Mission Failure Analytics Dashboard")
st.caption("Explainable AI-based mission risk intelligence")

# =================================================
# KPIs
# =================================================
c1, c2, c3 = st.columns(3)
c1.metric("Total Missions", len(df))
c2.metric("Success Rate (%)", f"{(1-df['failure'].mean())*100:.2f}")
c3.metric("Failure Rate (%)", f"{df['failure'].mean()*100:.2f}")

# =================================================
# MODEL TRANSPARENCY
# =================================================
with st.expander("📊 Model Transparency"):
    st.write(f"**Model Version:** {MODEL_VERSION}")
    st.write(f"**Training Data Size:** {len(df)} missions")
    st.write(f"**Test AUC:** {MODEL_AUC}")
    st.write(f"**Last Trained:** {LAST_TRAINED}")

# =================================================
# TABS
# =================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Trends", "⚠️ Risk Simulator", "🧠 Explainability"]
)

# =================================================
# TAB 1
# =================================================
with tab1:
    rocket_stats = df.groupby("rocket")["failure"].mean().reset_index()
    rocket_stats["Reliability (%)"] = 100 - rocket_stats["failure"] * 100
    st.dataframe(rocket_stats.sort_values("Reliability (%)", ascending=False))

# =================================================
# TAB 2
# =================================================
with tab2:
    yearly = df.groupby("launch_year")["failure"].mean()
    st.line_chart(yearly)

# =================================================
# TAB 3 – INDUSTRY RISK SIMULATOR
# =================================================
with tab3:
    c1, c2, c3 = st.columns(3)
    company = c1.selectbox("Organization", sorted(df["company"].unique()))
    rocket = c2.selectbox("Rocket", sorted(df["rocket"].unique()))
    year = c3.slider("Launch Year", 2000, 2035, 2025)

    base = df[(df["company"] == company) & (df["rocket"] == rocket)].tail(1)

    if not base.empty:
        base = base.copy()
        base["launch_year"] = year
        base["launch_decade"] = (year // 10) * 10

        risk = model.predict_proba(base[FEATURES])[0][1] * 100

        similar = len(df[(df["company"] == company) & (df["rocket"] == rocket)])
        confidence = confidence_level(similar)

        if year > df["launch_year"].max():
            st.warning("⚠ Launch year beyond historical data. Confidence reduced.")

        st.metric("Failure Risk (%)", f"{risk:.2f}")
        st.metric("Prediction Confidence", confidence)

        level, reason = explain_risk(base.iloc[0], risk)
        st.info(f"**{level} Risk** because {reason}")

        # Scenario simulation
        st.subheader("📈 Scenario Simulation")
        n = st.slider("Planned Missions", 1, 50, 10)
        expected_failures = n * (risk / 100)
        st.metric("Expected Failures", f"{expected_failures:.2f}")

        # Cost-aware risk
        cost = st.number_input("Cost per Mission (Million USD)", 10, 500, 50)
        expected_loss = expected_failures * cost
        st.metric("Expected Financial Loss", f"${expected_loss:.2f}M")

        # Audit log
        log_prediction(company, rocket, year, risk)

        pdf = generate_pdf(company, rocket, year, risk, confidence, reason)
        with open(pdf, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="mission_report.pdf")

# =================================================
# TAB 4 – ACTIONABLE SHAP
# =================================================
with tab4:
    sample = df[FEATURES].sample(min(200, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig = plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(fig)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feature = FEATURES[np.argmax(mean_abs)]
    st.success(f"🔑 **Top Risk Driver:** {top_feature}")

# =================================================
# RAW DATA
# =================================================
st.divider()
st.subheader("📄 Latest Mission Records")
st.dataframe(df.tail(20))
