"""
generate_data.py
================
Generates synthetic, realistic datasets for all three Mico Data Science
Intern projects. Run this once before running any analysis or Streamlit app.

Datasets created:
1. data/ecommerce_customer_behavior.csv  -> Project 1 (EDA)
2. data/house_prices.csv                  -> Project 2 (Regression)
3. data/churn_data.csv                    -> Project 3 (Classification)

Usage:
    python generate_data.py
"""

import os
import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Project 1: E-Commerce Customer Purchase Behavior (EDA)
# ---------------------------------------------------------------------------
def generate_ecommerce_data():
    n = 2000
    data_dir = ensure_data_folder("project_1_eda_customer_behavior")

    customer_id = [f"CUST{1000 + i}" for i in range(n)]
    age = np.random.normal(35, 10, n).astype(int)
    age = np.clip(age, 18, 75)
    gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    city_tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.3, 0.4, 0.3])
    membership = np.random.choice(["Basic", "Silver", "Gold", "Platinum"], n, p=[0.4, 0.3, 0.2, 0.1])

    base_spend = np.random.exponential(2000, n)
    purchase_amount = base_spend * np.array([{"Basic": 1.0, "Silver": 1.3, "Gold": 1.7, "Platinum": 2.2}[m] for m in membership])
    purchase_amount = purchase_amount * (1 + (age - 35) * 0.01)
    purchase_amount = np.round(purchase_amount, 2)

    purchase_frequency = np.random.poisson(3, n) + 1
    satisfaction = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.25, 0.4, 0.2])
    # cast to float so NaN (missing values) can be assigned later
    days_since_last = np.random.exponential(15, n).astype(float)
    missing_mask = np.random.choice([True, False], n, p=[0.08, 0.92])
    days_since_last[missing_mask] = np.nan

    outlier_idx = np.random.choice(n, 20, replace=False)
    purchase_amount[outlier_idx] *= np.random.uniform(5, 10, 20)

    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "City_Tier": city_tier,
        "Membership": membership,
        "Purchase_Amount": purchase_amount,
        "Purchase_Frequency": purchase_frequency,
        "Satisfaction_Score": satisfaction,
        "Days_Since_Last_Purchase": days_since_last,
    })

    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

    csv_path = os.path.join(data_dir, "ecommerce_customer_behavior.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated {csv_path} ({len(df)} rows)")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_data_folder(project_folder):
    data_dir = os.path.join(BASE_DIR, project_folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# ---------------------------------------------------------------------------
# Project 2: House Price Prediction (Regression)
# ---------------------------------------------------------------------------
def generate_house_data():
    n = 1500
    data_dir = ensure_data_folder("project_2_regression_house_prices")

    area_sqft = np.random.normal(1500, 400, n).astype(int)
    area_sqft = np.clip(area_sqft, 500, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    bathrooms = np.round(bedrooms * np.random.uniform(0.7, 1.0, n), 1)
    bathrooms = np.clip(bathrooms, 1, 5)
    age_of_house = np.random.exponential(15, n).astype(int)
    age_of_house = np.clip(age_of_house, 0, 100)

    location = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.4, 0.4, 0.2])
    # cast to float so NaN (missing values) can be assigned later
    bathrooms = bathrooms.astype(float)
    age_of_house = age_of_house.astype(float)
    price = (
        area_sqft * 2000
        + bedrooms * 150000
        + bathrooms * 100000
        - age_of_house * 5000
    ) * np.array([{"Urban": 1.3, "Suburban": 1.0, "Rural": 0.7}[l] for l in location])
    price = price * np.random.normal(1, 0.1, n)
    price = np.round(price, 0)

    missing_bath = np.random.choice(n, 50, replace=False)
    bathrooms[missing_bath] = np.nan
    missing_age = np.random.choice(n, 40, replace=False)
    age_of_house[missing_age] = np.nan

    df = pd.DataFrame({
        "Area_SqFt": area_sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_of_House": age_of_house,
        "Location": location,
        "Price": price,
    })

    csv_path = os.path.join(data_dir, "house_prices.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated {csv_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Project 3: Customer Churn Prediction (Classification)
# ---------------------------------------------------------------------------
def generate_churn_data():
    n = 3000
    data_dir = ensure_data_folder("project_3_classification_churn")

    customer_id = [f"CHURN{5000 + i}" for i in range(n)]
    tenure = np.random.exponential(18, n).astype(int)
    tenure = np.clip(tenure, 1, 72)
    monthly_charges = np.random.normal(65, 20, n).round(2)
    monthly_charges = np.clip(monthly_charges, 30, 150)
    total_charges = (monthly_charges * tenure * np.random.uniform(0.8, 1.2, n)).round(2)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.35, 0.2, 0.25, 0.2],
    )
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.4, 0.45, 0.15])
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.85, 0.15])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n, p=[0.4, 0.5, 0.1])

    churn_prob = 0.1
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0)
    churn_prob += np.where(tenure < 12, 0.15, 0)
    churn_prob += np.where(monthly_charges > 80, 0.1, 0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.05, 0)
    churn_prob += np.random.normal(0, 0.05, n)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob, n)

    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Contract": contract,
        "Payment_Method": payment_method,
        "Internet_Service": internet_service,
        "Gender": gender,
        "Senior_Citizen": senior_citizen,
        "Dependents": dependents,
        "Phone_Service": phone_service,
        "Multiple_Lines": multiple_lines,
        "Churn": churn,
    })

    missing_tc = np.random.choice(n, 60, replace=False)
    df.loc[missing_tc, "Total_Charges"] = np.nan

    csv_path = os.path.join(data_dir, "churn_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Generated {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    print("Generating synthetic datasets for Mico Data Science Intern projects...\n")
    generate_ecommerce_data()
    generate_house_data()
    generate_churn_data()
    print("\nAll datasets generated successfully!")
    print("Next step: run the analysis scripts or Streamlit apps in each project folder.")