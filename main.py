from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.train_model import train
from src.evaluate_model import evaluate

def run_pipeline():
    preprocess_data()
    feature_engineering()
    train()
    evaluate()

if __name__ == "__main__":
    run_pipeline()
