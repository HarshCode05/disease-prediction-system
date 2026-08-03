"""
preprocessing.py
-----------------
Reusable data cleaning and preprocessing functions shared across
the Heart Disease, Diabetes, and Kidney Disease pipelines.

Keeping this logic in one place means every notebook and training
script preprocesses data the same way, and the same functions can
be reused at inference time in the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset from the given path."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill missing values in numeric columns using the given strategy
    (median is generally safer than mean for skewed medical data).
    Categorical/object columns are filled with the most frequent value.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode the given categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, save_path: str = None):
    """
    Fit a StandardScaler on X_train and apply it to both train and test sets.
    Optionally saves the fitted scaler to disk so the same transform
    can be applied to live user input in the Streamlit app.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_path:
        joblib.dump(scaler, save_path)

    return X_train_scaled, X_test_scaled, scaler


def load_scaler(path: str):
    """Load a previously saved scaler (used during inference in the app)."""
    return joblib.load(path)


def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove rows where the given numeric columns have values outside
    the IQR-based bounds. Use carefully - in medical data, extreme
    values are sometimes real and clinically meaningful.
    """
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
