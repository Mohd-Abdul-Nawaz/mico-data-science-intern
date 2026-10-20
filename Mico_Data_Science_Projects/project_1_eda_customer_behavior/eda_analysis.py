"""
eda_analysis.py - Project 1: E-Commerce Customer Purchase Behavior (EDA)
========================================================================
Performs end-to-end exploratory data analysis on the synthetic e-commerce
dataset. Mirrors Mico JD responsibilities:
  - Collect, clean, preprocess, analyze structured datasets using Python
  - Perform EDA to identify patterns, trends, relationships, anomalies
  - Apply statistical concepts (correlation, hypothesis testing, p-values)
  - Handle missing values, outliers, categorical variables
  - Perform feature engineering on numerical, categorical, Boolean variables
  - Document analytical findings

Run: python eda_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ecommerce_customer_behavior.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")
    return df


def clean_data(df):
    """Handle missing values, duplicates, outliers."""
    df = df.copy()

    # Duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {dup_count} duplicate rows.")

    # Missing values
    missing = df.isnull().sum()
    print("\nMissing values before treatment:\n", missing[missing > 0])

    # Numerical: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical: fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values after treatment:\n", df.isnull().sum().sum())
    return df


def detect_outliers(df, column):
    """IQR-based outlier detection."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def perform_eda(df):
    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe(include="all"))

    print("\n===== CORRELATION MATRIX (Numerical) =====")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    print(corr)

    # Hypothesis test: Does membership affect purchase amount?
    print("\n===== HYPOTHESIS TEST: Membership vs Purchase Amount =====")
    groups = [g["Purchase_Amount"].values for _, g in df.groupby("Membership")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")
    print("Significant difference across membership tiers?" , "Yes" if p_val < 0.05 else "No")

    # Correlation: Age vs Purchase Amount
    age_purchase_corr = df["Age"].corr(df["Purchase_Amount"])
    print(f"\nCorrelation (Age vs Purchase_Amount): {age_purchase_corr:.4f}")

    # Satisfaction score distribution
    print("\n===== SATISFACTION SCORE DISTRIBUTION =====")
    print(df["Satisfaction_Score"].value_counts().sort_index())

    return corr


def feature_engineering(df):
    """Create new features from existing ones."""
    df = df.copy()

    # Boolean: High spender (above median purchase amount)
    median_spend = df["Purchase_Amount"].median()
    df["High_Spender"] = (df["Purchase_Amount"] > median_spend).astype(int)

    # Categorical binning: Age group
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 25, 35, 50, 100],
        labels=["Young (18-25)", "Adult (26-35)", "Middle (36-50)", "Senior (51+)"]
    )

    # Boolean: Recent customer (purchased within last 7 days)
    df["Recent_Customer"] = (df["Days_Since_Last_Purchase"] <= 7).astype(int)

    print("\n===== FEATURE ENGINEERING =====")
    print("High_Spender distribution:\n", df["High_Spender"].value_counts())
    print("Recent_Customer distribution:\n", df["Recent_Customer"].value_counts())

    return df


def save_outputs(df, corr):
    """Save cleaned data and correlation heatmap."""
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_ecommerce_data.csv"), index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap - E-Commerce Customer Behavior")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


def main():
    df = load_data()
    df = clean_data(df)
    corr = perform_eda(df)
    df = feature_engineering(df)
    save_outputs(df, corr)
    print("\nEDA complete. Check outputs/ folder for cleaned data and visualizations.")


if __name__ == "__main__":
    main()