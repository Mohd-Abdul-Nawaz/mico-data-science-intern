"""
streamlit_app.py - Project 2: Deployable House Price Predictor
==============================================================
Interactive Streamlit app for House Price Prediction using Linear Regression.
Reviewer can run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Mico DS Intern - Project 2: Regression", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "house_price_model.pkl")


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df["Bathrooms"] = df["Bathrooms"].fillna(df["Bathrooms"].median())
    df["Age_of_House"] = df["Age_of_House"].fillna(df["Age_of_House"].median())
    df = df.drop_duplicates()

    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Price"] >= Q1 - 1.5 * IQR) & (df["Price"] <= Q3 + 1.5 * IQR)]

    X = df[["Area_SqFt", "Bedrooms", "Bathrooms", "Age_of_House", "Location"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Area_SqFt", "Bedrooms", "Bathrooms", "Age_of_House"]),
            ("cat", OneHotEncoder(drop="first"), ["Location"]),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    # p = number of TRANSFORMED features (after one-hot encoding)
    p = model.named_steps["preprocessor"].get_feature_names_out().__len__()
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"r2": r2, "adj_r2": adj_r2, "mae": mae, "rmse": rmse}
    return model, metrics, X_test, y_test, y_pred


st.title("Mico Data Science Intern - Project 2")
st.subheader("House Price Prediction - Linear Regression")
st.markdown("""
Demonstrates **simple & multiple linear regression**, model evaluation with
**R², Adjusted R², MAE, RMSE**, and preprocessing of numerical + categorical variables.
""")

model, metrics, X_test, y_test, y_pred = train_model()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("R-squared", f"{metrics['r2']:.3f}")
col2.metric("Adjusted R²", f"{metrics['adj_r2']:.3f}")
col3.metric("MAE", f"₹{metrics['mae']:,.0f}")
col4.metric("RMSE", f"₹{metrics['rmse']:,.0f}")

st.divider()

# Prediction form
st.header("Predict House Price")
st.markdown("Enter house details to get a price estimate:")

col_a, col_b = st.columns(2)
with col_a:
    area = st.number_input("Area (Sq Ft)", min_value=500, max_value=5000, value=1500, step=50)
    bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=2)
with col_b:
    age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=10, step=1)
    location = st.selectbox("Location", options=["Urban", "Suburban", "Rural"], index=1)

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "Area_SqFt": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_of_House": age,
        "Location": location,
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: **₹{prediction:,.0f}**")
    st.info(f"Model R² on test data: {metrics['r2']:.3f} | RMSE: ₹{metrics['rmse']:,.0f}")

st.divider()

# Actual vs Predicted
st.header("Model Performance: Actual vs Predicted")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted House Prices")
st.pyplot(fig)

st.divider()
st.markdown("*Built for Mico Data Science Intern application | Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit*")