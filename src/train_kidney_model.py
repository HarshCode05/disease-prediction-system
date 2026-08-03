"""
train_kidney_model.py
-----------------------
Clean, script-form training pipeline for the Chronic Kidney Disease model.
Production version of notebooks/06_kidney_models.ipynb.

Run with: python train_kidney_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_data, handle_missing_values, encode_categorical, scale_features

RANDOM_STATE = 42
DATA_PATH = "../data/kidney_disease.csv"
MODEL_SAVE_PATH = "../models/kidney_model.h5"
SCALER_SAVE_PATH = "../models/kidney_scaler.pkl"
COLUMNS_SAVE_PATH = "../models/kidney_columns.pkl"  # needed to align app input with training columns


def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", np.nan)
    df = df.replace("nan", np.nan)

    for col in ["pcv", "wc", "rc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


def main():
    np.random.seed(RANDOM_STATE)

    df = load_data(DATA_PATH)
    df = clean_raw_data(df)

    target_col = "classification" if "classification" in df.columns else df.columns[-1]
    df[target_col] = df[target_col].astype(str).str.strip()
    df[target_col] = df[target_col].replace({"ckd\t": "ckd"})
    y = df[target_col].map({"ckd": 1, "notckd": 0})

    df_features = df.drop(columns=[target_col])
    df_features = handle_missing_values(df_features)

    cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    X = encode_categorical(df_features, cat_cols)

    # Save the final column order/names - needed so the Streamlit app can
    # build a matching input row at inference time
    import joblib
    joblib.dump(X.columns.tolist(), COLUMNS_SAVE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train, X_test, save_path=SCALER_SAVE_PATH
    )

    model = build_ann(X_train_scaled.shape[1])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
