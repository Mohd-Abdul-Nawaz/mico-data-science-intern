# Deployment Guide — Mico Data Science Intern Projects

This guide shows how to run, test, and deploy the three data science projects so a reviewer can see your work.

---

## Project Structure

```
Mico_Data_Science_Projects/
├── generate_data.py                  # Creates all synthetic datasets
├── requirements.txt                  # Python dependencies (bounded version ranges)
├── DEPLOYMENT_GUIDE.md               # This file
├── _run_all.py                       # Runs all 3 analysis scripts, prints key metrics
├── _verify_apps.py                   # Headless Streamlit smoke test for all 3 apps
├── .gitignore
│
├── project_1_eda_customer_behavior/  # Project 1: EDA
│   ├── data/
│   │   └── ecommerce_customer_behavior.csv
│   ├── outputs/
│   │   ├── cleaned_ecommerce_data.csv
│   │   └── correlation_heatmap.png
│   ├── eda_analysis.py               # Analysis script
│   └── streamlit_app.py              # Deployable dashboard
│
├── project_2_regression_house_prices/ # Project 2: Regression
│   ├── data/
│   │   └── house_prices.csv
│   ├── outputs/
│   │   ├── regression_results.png
│   │   └── predictions.csv
│   ├── regression_model.py           # Analysis script
│   └── streamlit_app.py              # Deployable app
│
└── project_3_classification_churn/   # Project 3: Classification
    ├── data/
    │   └── churn_data.csv
    ├── outputs/
    │   ├── classification_results.png
    │   └── predictions.csv
    ├── classification_model.py       # Analysis script
    └── streamlit_app.py              # Deployable app
```

---

## Step 0: Install Dependencies

Open a terminal/command prompt in the `Mico_Data_Science_Projects` folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, scipy, imbalanced-learn, matplotlib,
seaborn, streamlit, pyarrow, joblib, plotly, openpyxl, nbformat.

> **Version note:** `requirements.txt` uses **bounded ranges** (e.g. `numpy>=2.1,<2.3`)
> instead of exact pins, so pip can always find a **prebuilt wheel** on Python 3.11, 3.12
> *and* 3.13 (Streamlit Cloud's default). The exact pins used in an earlier draft
> (numpy 1.26.4, scipy 1.13.1, matplotlib 3.8.4, pyarrow 17.0.0) ship **no wheels for
> Python >= 3.13**, so pip falls back to compiling them from source and the cloud build
> hangs forever at `Preparing metadata (pyproject.toml): still running...`.
>
> Two constraints are kept on purpose inside the ranges:
> - `pandas < 3.0` — seaborn 0.13's `boxplot` relies on `SeriesGroupBy.grouper`, removed
>   in pandas >= 3.0. Project 1's dashboard therefore also draws its boxplots with
>   matplotlib directly (`boxplot_by()`), so it works on any version.
> - `pyarrow >= 18` (first release with Python 3.13 wheels) — which requires `numpy >= 2.1`,
>   so numpy is bounded to `>= 2.1` as well.
>
> Re-running `_run_all.py` and `_verify_apps.py` on this stack reproduces every metric
> in the Verified Metrics table below exactly.

---

## Step 1: Generate the Datasets

The projects use synthetic (computer-generated) datasets so you don't need to find external data. Run once:

```bash
python generate_data.py
```

This creates three CSV files, one per project, with realistic missing values, outliers, and patterns.

Expected output:

```
[OK] Generated ...\project_1_eda_customer_behavior\data\ecommerce_customer_behavior.csv (2010 rows)
[OK] Generated ...\project_2_regression_house_prices\data\house_prices.csv (1500 rows)
[OK] Generated ...\project_3_classification_churn\data\churn_data.csv (3000 rows)
```

---


## Step 2: Run Each Project

### Option A: Run the Analysis Script (generates outputs/plots)

```bash
# Project 1
cd project_1_eda_customer_behavior
python eda_analysis.py

# Project 2
cd project_2_regression_house_prices
python regression_model.py

# Project 3
cd project_3_classification_churn
python classification_model.py
```

Each script prints metrics to the console and saves cleaned data + plots to its `outputs/` folder.

### Option B: Run the Streamlit App (interactive, reviewer-friendly)

Streamlit apps let a reviewer interact with your models in a browser — ideal for showing your work.

```bash
# Project 1
cd project_1_eda_customer_behavior
streamlit run streamlit_app.py

# Project 2
cd project_2_regression_house_prices
streamlit run streamlit_app.py

# Project 3
cd project_3_classification_churn
streamlit run streamlit_app.py
```

After running, Streamlit gives you a local URL (usually `http://localhost:8501`). Open it in a browser to interact with the dashboard.

---

## Step 3: Validate Everything (Automated)

Two helper scripts are included so you can prove the projects work before sharing them:

```bash
# Runs all three analysis scripts and prints the key metrics
python _run_all.py

# Executes each Streamlit app headlessly and asserts zero uncaught exceptions
python _verify_apps.py
```

Expected output:

```
### project_1_eda_customer_behavior\eda_analysis.py
    ANOVA F-statistic: 18.7017, p-value: 0.0000
    Correlation (Age vs Purchase_Amount): 0.0499
### project_2_regression_house_prices\regression_model.py
    R-squared:        0.8657
    Adjusted R-squared: 0.8629
    MAE:              345,060
    RMSE:             441,571
### project_3_classification_churn\classification_model.py
    accuracy                           0.63       600
    ROC-AUC: 0.6840
RESULT: 3/3 scripts succeeded
```

```
SUMMARY: 3/3 apps passed
```

> `_verify_apps.py` uses Streamlit's `AppTest` harness, so it exercises the real app
> code (not just an import) and surfaces runtime errors such as plotting API changes.

---

## Step 4: Deploy Online (Free) — So a Reviewer Can See It Without Installing Anything

Use **Streamlit Community Cloud** (free) to host your apps. A reviewer just needs a URL.

### Deploy via Streamlit Cloud

1. Push this folder to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub, click "New app", select your repository.
4. Set the main file path to one of the `streamlit_app.py` files (e.g., `project_2_regression_house_prices/streamlit_app.py`).
5. Click "Deploy". Streamlit gives you a public URL like `https://your-app.streamlit.app`.

**Tip:** Deploy all three apps as separate Streamlit Cloud apps, or create a single multi-page Streamlit app combining all three.

> **Streamlit Cloud notes — read these if the build hangs:**
> - Streamlit Cloud installs dependencies **only from a `requirements.txt` at the repo root**.
>   The root copy is mandatory — without it you get `ModuleNotFoundError: matplotlib` because
>   nothing was installed (the nested file alone is ignored).
> - `runtime.txt` (`python-3.11`) and `.python-version` (`3.11`) force Python 3.11, matching the
>   local environment. If the platform ignores them and falls back to its default 3.13, the
>   bounded ranges still install from wheels — either way the build completes instead of hanging.
> - Keep the bounded ranges in `requirements.txt`. Do **not** switch back to exact pins (no
>   Python 3.13 wheels) and do **not** let pandas reach 3.0 (seaborn boxplot break).

### Deploy via Render (alternative free option)

1. Push code to GitHub.
2. Go to [https://render.com](https://render.com), create a "Web Service".
3. Connect your GitHub repo.
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run project_2_regression_house_prices/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

## What Each Project Demonstrates (Mico JD Alignment)

| Project | Focus | Mico JD Keywords Covered |
|---------|-------|-------------------------|
| **Project 1: E-Commerce EDA** | Exploratory Data Analysis | Data cleaning, missing values, outliers, EDA, correlation, hypothesis testing (p-values), feature engineering, categorical/Boolean variables, documentation |
| **Project 2: House Price Regression** | Linear Regression | Simple & multiple linear regression, train/test split, preprocessing, R², Adjusted R², MAE, MSE, RMSE, model evaluation |
| **Project 3: Customer Churn Classification** | Logistic Regression | Logistic regression, class imbalance (SMOTE), categorical/Boolean encoding, confusion matrix, ROC-AUC, precision/recall/F1 |

---

## Verified Metrics (safe to quote in your resume)

| Project | Metric | Value |
|---------|--------|-------|
| 1 — EDA | ANOVA F-statistic (Membership vs Purchase Amount) | 18.70 (p < 0.001) |
| 1 — EDA | Pearson correlation (Age vs Purchase Amount) | 0.050 |
| 2 — Regression | R² | 0.866 |
| 2 — Regression | Adjusted R² (computed on post-encoding feature count) | 0.863 |
| 2 — Regression | MAE / RMSE | ₹345,060 / ₹441,571 |
| 3 — Classification | ROC-AUC | 0.684 |
| 3 — Classification | Churn-class recall | 0.68 (vs 0.61 for retained class) |

---


## How to Present These in Your Resume

In the **Projects** section of your resume, add GitHub/Streamlit links:

```
### E-Commerce Customer Purchase Behavior Analysis
Python | Pandas | NumPy | SciPy | Matplotlib | Seaborn | Streamlit
- Performed EDA on 2,010 customer records; handled missing values, outliers, duplicates
- Applied ANOVA hypothesis testing (F = 18.70, p < 0.001) and correlation analysis
- Engineered Boolean & categorical features (High_Spender, Recent_Customer, Age_Group)
- [Live Demo](https://mico-data-science-intern-kte8yj6a7aaryhfnj3oyab.streamlit.app/) | [Code](https://github.com/Mohd-Abdul-Nawaz/mico-data-science-intern/tree/main/project_1_eda_customer_behavior)
```

```
### House Price Prediction — Multiple Linear Regression
Python | Scikit-learn | Pandas | Matplotlib | Streamlit
- Built Multiple Linear Regression on 1,494 records (R² = 0.866, Adjusted R² = 0.863, RMSE = ₹441,571)
- Preprocessed numerical + categorical variables; handled missing data & outliers
- Evaluated with R², Adjusted R², MAE, MSE, RMSE; residual analysis
- [Live Demo](https://mico-data-science-intern-uz5cau86ztv5kmua6r9jbv.streamlit.app/) | [Code](https://github.com/Mohd-Abdul-Nawaz/mico-data-science-intern/tree/main/project_2_regression_house_prices)
```

```
### Customer Churn Prediction — Logistic Regression with SMOTE
Python | Scikit-learn | Imbalanced-learn | Streamlit
- Built Logistic Regression to predict churn (ROC-AUC = 0.684)
- Applied SMOTE after encoding, inside an imbalanced-learn pipeline
- Evaluated with confusion matrix, precision, recall, F1-score, ROC curve
- [Live Demo](https://mico-data-science-intern-hcq56yhkqpkmgjaf9clrhz.streamlit.app/) | [Code](https://github.com/Mohd-Abdul-Nawaz/mico-data-science-intern/tree/main/project_3_classification_churn)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `streamlit` command not found | Run `pip install streamlit` or use `python -m streamlit run ...` |
| ModuleNotFoundError: sklearn | Run `pip install scikit-learn` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| SMOTE error | Run `pip install imbalanced-learn` |
| `ValueError: cannot convert float NaN to integer` | You are running an older `generate_data.py`; regenerate the datasets with the current script |
| Boxplot crashes with `UnboundLocalError: boxprops` | seaborn 0.13 vs pandas >= 3.0 conflict — install the bounded versions from `requirements.txt` (`pandas < 3.0`), or use the matplotlib-based `boxplot_by()` already in Project 1 |
| Streamlit Cloud build hangs at `Preparing metadata (pyproject.toml): still running...` | pip is compiling a package from source — the Python version has no wheel for it. Ensure the **root** `requirements.txt` with the bounded ranges is committed; `runtime.txt` / `.python-version` set 3.11, and the ranges install from wheels even on 3.13. |
| App shows old data | Delete `outputs/` folder and re-run the analysis script first |

---

## Next Steps

1. Run `python generate_data.py` to create datasets
2. Run `python _run_all.py` and `python _verify_apps.py` to confirm everything passes
3. Run each `streamlit_app.py` to verify locally in the browser
4. Push to GitHub and deploy on Streamlit Cloud
5. Add the live demo URLs to your resume
6. Be ready to explain every plot, metric, and preprocessing step in the interview

Good luck with your Mico Data Science Intern application!
