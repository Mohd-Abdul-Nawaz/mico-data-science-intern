"""
streamlit_app.py - Project 1: Deployable EDA Dashboard
======================================================
Interactive Streamlit app for E-Commerce Customer Purchase Behavior analysis.
Reviewer can run:  streamlit run streamlit_app.py
Or deploy to Streamlit Cloud for free.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

st.set_page_config(page_title="Mico DS Intern - Project 1: EDA", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ecommerce_customer_behavior.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_ecommerce_data.csv")


def boxplot_by(data, x_col, y_col, title, palette="Set2"):
    """Grouped boxplot that works across seaborn/pandas versions.

    seaborn's boxplot relies on `SeriesGroupBy.grouper`, which pandas >= 3.0
    removed, so we draw the boxes with matplotlib directly. This keeps the
    chart identical while removing a hard dependency on seaborn internals.
    """
    groups = list(data.groupby(x_col, observed=True))
    labels = [str(name) for name, _ in groups]
    values = [grp[y_col].dropna().to_numpy() for _, grp in groups]

    fig, ax = plt.subplots()
    # matplotlib renamed `labels` -> `tick_labels` in 3.9; support both.
    try:
        bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.6)
    except TypeError:
        bp = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.6)

    # Colour the boxes using a light palette without relying on seaborn.
    base_colors = plt.get_cmap("tab10").colors
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(base_colors[i % len(base_colors)])
        box.set_alpha(0.65)
    for whisker in bp["whiskers"]:
        whisker.set_color("gray")
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    return fig, ax


@st.cache_data
def load_data():
    if os.path.exists(CLEAN_PATH):
        return pd.read_csv(CLEAN_PATH)
    return pd.read_csv(DATA_PATH)


st.title("Mico Data Science Intern - Project 1")
st.subheader("E-Commerce Customer Purchase Behavior - Exploratory Data Analysis")
st.markdown("""
This dashboard mirrors Mico's Data Science Intern responsibilities:
**data cleaning, EDA, statistical testing, and feature engineering** on structured datasets.
""")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
gender_filter = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
membership_filter = st.sidebar.multiselect("Membership", options=df["Membership"].unique(), default=df["Membership"].unique())
city_tier_filter = st.sidebar.multiselect("City Tier", options=df["City_Tier"].unique(), default=df["City_Tier"].unique())

filtered = df[
    (df["Gender"].isin(gender_filter)) &
    (df["Membership"].isin(membership_filter)) &
    (df["City_Tier"].isin(city_tier_filter))
]

st.write(f"Showing {len(filtered)} records after filtering.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Purchase Amount", f"₹{filtered['Purchase_Amount'].mean():,.0f}")
col2.metric("Avg Age", f"{filtered['Age'].mean():.1f}")
col3.metric("Avg Satisfaction", f"{filtered['Satisfaction_Score'].mean():.2f} / 5")
col4.metric("Missing Values", int(filtered.isnull().sum().sum()))

st.divider()

# Distribution plots
st.header("1. Distributions")
col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots()
    sns.histplot(filtered["Purchase_Amount"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_title("Purchase Amount Distribution")
    st.pyplot(fig)
with col_b:
    fig, ax = plt.subplots()
    sns.countplot(data=filtered, x="Satisfaction_Score", hue="Satisfaction_Score", legend=False, ax=ax, palette="viridis")
    ax.set_title("Satisfaction Score Distribution")
    st.pyplot(fig)

st.divider()

# Categorical analysis
st.header("2. Purchase Amount by Category")
col_c, col_d = st.columns(2)
with col_c:
    fig, ax = boxplot_by(filtered, "Membership", "Purchase_Amount", "Purchase Amount by Membership")
    st.pyplot(fig)
with col_d:
    fig, ax = boxplot_by(filtered, "City_Tier", "Purchase_Amount", "Purchase Amount by City Tier")
    st.pyplot(fig)

st.divider()

# Correlation
st.header("3. Correlation Analysis")
num_cols = filtered.select_dtypes(include=[np.number]).columns
corr = filtered[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

st.markdown("""
**Key Insight:** Age shows a weak correlation with Purchase Amount, while Membership tier
shows a strong positive relationship. Higher membership tiers correlate with higher spending.
""")

# Hypothesis test
st.header("4. Hypothesis Testing (ANOVA)")
st.markdown("**Question:** Does Purchase Amount differ significantly across Membership tiers?")
groups = [g["Purchase_Amount"].values for _, g in filtered.groupby("Membership")]
f_stat, p_val = stats.f_oneway(*groups)
st.write(f"F-statistic: `{f_stat:.4f}`")
st.write(f"p-value: `{p_val:.4f}`")
if p_val < 0.05:
    st.success("Result: Statistically significant (p < 0.05). Membership tier affects purchase amount.")
else:
    st.info("Result: Not statistically significant.")

st.divider()

# Feature engineering preview
st.header("5. Feature Engineering Preview")
st.markdown("Derived Boolean and categorical features from raw variables.")
med = df["Purchase_Amount"].median()
df_preview = df.copy()
df_preview["High_Spender"] = (df_preview["Purchase_Amount"] > med).astype(int)
df_preview["Recent_Customer"] = (df_preview["Days_Since_Last_Purchase"] <= 7).astype(int)
st.dataframe(df_preview[["CustomerID", "Purchase_Amount", "High_Spender", "Days_Since_Last_Purchase", "Recent_Customer"]].head(10))

st.divider()
st.markdown("*Built for Mico Data Science Intern application | Stack: Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Streamlit*")