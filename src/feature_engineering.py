import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import PROCESSED_PATH

def feature_engineering():
    df = pd.read_csv(PROCESSED_PATH)

    # Rocket reliability
    df['rocket_failure_rate'] = df.groupby('rocket')['failure'].transform('mean')
    df['rocket_launch_count'] = df.groupby('rocket')['failure'].transform('count')

    # Organization failure rate
    df['org_failure_rate'] = df.groupby('company')['failure'].transform('mean')

    # Historical risk feature
    df = df.sort_values('launch_date')
    df['past_failure_rate'] = (
        df.groupby('company')['failure']
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['past_failure_rate'] = df['past_failure_rate'].fillna(df['failure'].mean())

    # Encoding
    for col in ['company', 'location', 'rocket']:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    df.to_csv(PROCESSED_PATH, index=False)
    return df

if __name__ == "__main__":
    feature_engineering()
