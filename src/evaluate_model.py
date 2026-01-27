import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import PROCESSED_PATH, MODEL_PATH

def evaluate():
    df = pd.read_csv(PROCESSED_PATH)
    model = joblib.load(MODEL_PATH)

    features = [
        'launch_year',
        'launch_decade',
        'rocket_failure_rate',
        'rocket_launch_count',
        'org_failure_rate',
        'past_failure_rate',
        'company_enc',
        'location_enc',
        'rocket_enc'
    ]

    X = df[features]
    y = df['failure']

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print("Accuracy:", accuracy_score(y, preds))
    print("ROC AUC:", roc_auc_score(y, probs))

if __name__ == "__main__":
    evaluate()
