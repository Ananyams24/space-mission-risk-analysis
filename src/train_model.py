import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_PATH, MODEL_PATH, RANDOM_STATE
import os

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train():
    df = pd.read_csv(PROCESSED_PATH)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    return X_test, y_test

if __name__ == "__main__":
    train()
