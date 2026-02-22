import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_preprocess(filepath):
    """
    Load and preprocess the US Medical Cost Personal Dataset.
    Steps:
    1. Load the CSV
    2. Check for missing values and duplicates
    3. Encode categorical features (sex, smoker, region)
    4. Normalize using StandardScaler
    5. Split 75% train, 25% test
    """
    df = pd.read_csv(filepath)

    # Check missing values
    print("Missing values:\n", df.isnull().sum())

    # Check duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Encode categorical features
    df_encoded = df.copy()
    df_encoded['sex'] = df_encoded['sex'].map({'male': 1, 'female': 0})
    df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
    df_encoded = pd.get_dummies(df_encoded, columns=['region'], drop_first=True, dtype=int)

    # Separate features and target
    X = df_encoded.drop(columns=['charges'])
    y = df_encoded['charges']

    # Standard Scaler normalization
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 75/25 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test, df, scaler
