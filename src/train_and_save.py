"""Train all models and save them + scaler for the Streamlit app."""
import os
import sys
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_and_preprocess
from src.models import train_all_models

def main():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'insurance.csv')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, df, scaler = load_and_preprocess(data_path)

    print("\nTraining all models (this may take several minutes)...")
    results = train_all_models(X_train, y_train)

    # Save each model
    for name, res in results.items():
        model_path = os.path.join(models_dir, f'{name}_model.pkl')
        joblib.dump(res['model'], model_path)
        print(f"Saved {name} model to {model_path}")

    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(list(X_train.columns), os.path.join(models_dir, 'feature_names.pkl'))
    print(f"\nSaved scaler and feature names to {models_dir}")
    print("Done! You can now run the Streamlit app: streamlit run app.py")

if __name__ == '__main__':
    main()
