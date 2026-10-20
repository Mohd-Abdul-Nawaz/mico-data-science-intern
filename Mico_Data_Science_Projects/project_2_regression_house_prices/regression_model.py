"""
regression_model.py - Project 2: House Price Prediction (Regression)
====================================================================
Mirrors Mico JD responsibilities:
  - Build understanding of linear regression, logistic regression fundamentals
  - Evaluate regression models using R-squared, Adjusted R-squared, MAE, MSE, RMSE
  - Handle missing values, outliers, categorical variables
  - Perform feature engineering for numerical & categorical variables
  - Document analytical findings

Run: python regression_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()

    # Missing value treatment
    df["Bathrooms"] = df["Bathrooms"].fillna(df["Bathrooms"].median())
    df["Age_of_House"] = df["Age_of_House"].fillna(df["Age_of_House"].median())

    # Remove duplicates
    df = df.drop_duplicates()

    # Outlier capping on Price (IQR method)
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Price"] >= Q1 - 1.5 * IQR) & (df["Price"] <= Q3 + 1.5 * IQR)]

    print(f"After preprocessing: {df.shape}")
    return df


def build_model(X_train, X_test, y_train, y_test):
    numeric_features = ["Area_SqFt", "Bedrooms", "Bathrooms", "Age_of_House"]
    categorical_features = ["Location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)

    # Adjusted R-squared
    # p must be the number of TRANSFORMED features (after one-hot encoding),
    # not the number of raw columns in X_test.
    n = len(y_test)
    p = model.named_steps["preprocessor"].get_feature_names_out().__len__()
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"R-squared:        {r2:.4f}")
    print(f"Adjusted R-squared: {adj_r2:.4f}")
    print(f"MAE:              {mae:,.0f}")
    print(f"MSE:              {mse:,.0f}")
    print(f"RMSE:             {rmse:,.0f}")

    return model, y_pred, {"r2": r2, "adj_r2": adj_r2, "mae": mae, "mse": mse, "rmse": rmse}


def plot_results(y_test, y_pred, metrics):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color="seagreen")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "regression_results.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR}/regression_results.png")


def main():
    df = load_data()
    df = preprocess(df)

    X = df[["Area_SqFt", "Bedrooms", "Bathrooms", "Age_of_House", "Location"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, y_pred, metrics = build_model(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, metrics)

    # Save predictions
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"\nPredictions saved to {OUTPUT_DIR}/predictions.csv")
    print("\nRegression complete.")


if __name__ == "__main__":
    main()