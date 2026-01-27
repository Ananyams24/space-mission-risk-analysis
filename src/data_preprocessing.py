import pandas as pd
from src.config import DATA_PATH, PROCESSED_PATH

def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # ✅ FIX: clean column names
    df.columns = df.columns.str.strip()

    # Select required columns
    df = df[['Company Name', 'Location', 'Rocket', 'Status Mission', 'Datum']]

    # Rename columns
    df.columns = ['company', 'location', 'rocket', 'mission_status', 'launch_date']

    # Failure flag
    df['failure'] = (df['mission_status'] != 'Success').astype(int)

    # Date processing
    df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
    df['launch_year'] = df['launch_date'].dt.year
    df['launch_decade'] = (df['launch_year'] // 10) * 10

    # Handle missing values
    df['rocket'] = df['rocket'].fillna('Unknown')
    df.dropna(subset=['launch_date', 'company', 'location'], inplace=True)

    # Save processed file
    df.to_csv(PROCESSED_PATH, index=False)

    return df

if __name__ == "__main__":
    preprocess_data()
