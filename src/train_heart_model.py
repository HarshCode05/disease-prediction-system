"""
train_heart_model.py
---------------------
Clean, script-form training pipeline for the Heart Disease model.
Production version of notebooks/04_heart_models.ipynb.

Updated for the actual dataset format: text categorical columns
(cp, restecg, slope, thal, sex, fbs, exang), an id/dataset column
that gets dropped, and a `num` (0-4) target converted to binary.

Run with: python train_heart_model.py
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_data, handle_missing_values, encode_categorical, scale_features

RANDOM_STATE = 42
DATA_PATH = "../data/heart_disease.csv"
MODEL_SAVE_PATH = "../models/heart_model.h5"
SCALER_SAVE_PATH = "../models/heart_scaler.pkl"
COLUMNS_SAVE_PATH = "../models/heart_columns.pkl"


def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    np.random.seed(RANDOM_STATE)

    df = load_data(DATA_PATH)
    df = df.drop(columns=["id", "dataset"])

    y = (df["num"] > 0).astype(int)
    df_features = df.drop(columns=["num"])

    df_features = handle_missing_values(df_features)

    cat_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    X = encode_categorical(df_features, cat_cols)

    # Save column order - needed so the Streamlit app builds a matching
    # input row (same one-hot columns, same order) at inference time
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
