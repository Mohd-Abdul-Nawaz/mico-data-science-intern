"""
classification_model.py - Project 3: Customer Churn Prediction (Classification)
===============================================================================
Mirrors Mico JD responsibilities:
  - Build understanding of logistic regression fundamentals
  - Handle class imbalance, categorical & Boolean variables
  - Evaluate classification models
  - Document analytical approach and justify decisions

Run: python classification_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()

    # Missing Total_Charges -> fill with median
    df["Total_Charges"] = df["Total_Charges"].fillna(df["Total_Charges"].median())

    # Drop duplicates
    df = df.drop_duplicates()

    print(f"After preprocessing: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    return df


def build_model(X_train, X_test, y_train, y_test):
    numeric_features = ["Tenure", "Monthly_Charges", "Total_Charges", "Senior_Citizen"]
    categorical_features = [
        "Contract", "Payment_Method", "Internet_Service",
        "Gender", "Dependents", "Phone_Service", "Multiple_Lines"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # SMOTE is applied AFTER the preprocessor so only numeric, encoded features
    # are oversampled. Applying SMOTE to raw string columns is invalid.
    model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    return model, y_pred, y_prob, cm, roc_auc


def plot_results(cm, y_test, y_prob):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_results.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {OUTPUT_DIR}/classification_results.png")


def main():
    df = load_data()
    df = preprocess(df)

    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class imbalance is handled inside the pipeline (preprocess -> SMOTE -> model),
    # which guarantees SMOTE is applied only to encoded numeric training features.
    model, y_pred, y_prob, cm, roc_auc = build_model(X_train, X_test, y_train, y_test)
    plot_results(cm, y_test, y_prob)

    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"\nPredictions saved to {OUTPUT_DIR}/predictions.csv")
    print("\nClassification complete.")


if __name__ == "__main__":
    main()