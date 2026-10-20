"""
streamlit_app.py - Project 3: Deployable Customer Churn Predictor
================================================================
Interactive Streamlit app for Customer Churn Prediction using Logistic Regression.
Reviewer can run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(page_title="Mico DS Intern - Project 3: Classification", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_data.csv")


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df["Total_Charges"] = df["Total_Charges"].fillna(df["Total_Charges"].median())
    df = df.drop_duplicates()

    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    # SMOTE runs AFTER preprocessing, inside the pipeline, so only encoded
    # numeric features are oversampled (SMOTE on raw strings is invalid).
    model = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return model, cm, roc_auc, y_test, y_prob, y_pred


st.title("Mico Data Science Intern - Project 3")
st.subheader("Customer Churn Prediction - Logistic Regression")
st.markdown("""
Demonstrates **logistic regression**, **class imbalance handling (SMOTE)**,
and evaluation with **confusion matrix, ROC-AUC, precision, recall, F1-score**.
Relevant to Mico's subscription-based services (LINE, SMS/RCS, Voice AI).
""")

model, cm, roc_auc, y_test, y_prob, y_pred = train_model()

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc:.3f}")
col2.metric("Accuracy", f"{(y_pred == y_test).mean():.3f}")
col3.metric("Churn Rate (Actual)", f"{y_test.mean():.2%}")

st.divider()

st.header("Predict Customer Churn")
st.markdown("Enter customer details to predict churn probability:")

col_a, col_b = st.columns(2)
with col_a:
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=30, max_value=150, value=70)
    total_charges = st.number_input("Total Charges (₹)", min_value=0, max_value=10000, value=700)
    senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col_b:
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], index=0)
    payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer", "Credit card"], index=0)
    internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=1)
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)

dependents = st.selectbox("Dependents", options=["Yes", "No"], index=1)
phone_service = st.selectbox("Phone Service", options=["Yes", "No"], index=0)
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"], index=1)

if st.button("Predict Churn", type="primary"):
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Senior_Citizen": senior_citizen,
        "Contract": contract,
        "Payment_Method": payment_method,
        "Internet_Service": internet_service,
        "Gender": gender,
        "Dependents": dependents,
        "Phone_Service": phone_service,
        "Multiple_Lines": multiple_lines,
    }])
    prob = model.predict_proba(input_df)[0][1]
    pred = "Will Churn" if prob >= 0.5 else "Will Stay"

    if prob >= 0.5:
        st.error(f"Prediction: **{pred}** | Churn Probability: **{prob:.2%}**")
    else:
        st.success(f"Prediction: **{pred}** | Churn Probability: **{prob:.2%}**")

st.divider()

st.header("Model Evaluation")
col_x, col_y = st.columns(2)
with col_x:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
with col_y:
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.divider()
st.markdown("*Built for Mico Data Science Intern application | Stack: Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn, Streamlit*")