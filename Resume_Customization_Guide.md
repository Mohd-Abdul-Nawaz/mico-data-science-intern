# Resume Customization Guide — Mico Data Science Intern

Use this guide to fill in the placeholders in `Mico_Data_Science_Intern_Resume.md` and turn it into a selection-ready resume.

> **Status of this copy:** The header, career objective, education, skills, achievements and
> declaration sections are filled in with real details, and the **PROJECTS** section (§5)
> carries the three verified projects with live demo links. Certifications, a LinkedIn URL and
> any placeholder lines are deliberately **omitted** — include a section only when you have a
> truthful, defensible entry for it.

---

## 1. Header — Personal Details

| Field | Value in this file |
|-------|--------------------|
| Name | `Mohd Abdul Nawaz` |
| Location | `Guwahati, Assam, India` |
| Phone | `(+91) 8247024362` |
| Email | `nawaziiitguwahati@gmail.com` |
| GitHub | `github.com/Mohd-Abdul-Nawaz` |
| Relocation | `Open to relocation: Bangalore, Karnataka` |

> **Tip:** Since Mico is based in Bangalore, the **"Open to relocation: Bangalore, Karnataka"**
> line reassures recruiters. The live demo links already sit in the PROJECTS section, so no
> separate portfolio link is needed.
>
> **Note:** This copy deliberately includes **no LinkedIn URL** and no placeholder links of any
> kind — never publish a placeholder. Add a real LinkedIn URL only if you create the profile
> and choose to include it.

---

## 2. Career Objective — Make It Specific

- The branch (`Computer Science & Engineering`) and university (`IIIT Guwahati`) are already filled in — keep them accurate.
- Keep the mention of **LINE, SMS/RCS, Voice AI, and web** — it shows you researched Mico and understand their product (conversational layer for Japanese brands).
- Do **not** exaggerate experience. Since you're a fresher, emphasize academic projects, eagerness to learn, and alignment with Mico's values (Wow the Customer, Invest in Passion, Beyond Borders, Be the Change).

---

## 3. Education

- **CGPA:** CPI 7.30 / 10 is already entered. If yours differs, update it; if it's below 7.0, you may omit it or write "Pursuing."
- **Relevant Coursework:** Keep only courses you've actually taken or are confident discussing in an interview.
- **Class 12 / Class 10:** Already filled (S.T. George's Grammar Boys School 94.25%, Narayana Junior College 88.9%). Verify these match your records.

---

## 4. Technical Skills — Match the JD Keywords

Mico's JD specifically mentions these. Keep them if you can honestly discuss them:

- Python, Pandas, NumPy, Scikit-learn
- SQL, Excel
- EDA, data cleaning, missing value treatment, outlier detection, feature engineering
- Categorical, numerical, Boolean variables
- Hypothesis testing, p-values, correlation analysis
- Simple & multiple linear regression, logistic regression
- R², Adjusted R², MAE, MSE, RMSE
- Overfitting, model validation, class imbalance, decision trees
- Matplotlib, Seaborn
- Jupyter Notebook, Google Colab

> Remove any skill you cannot explain confidently. Interviewers will ask about everything listed here.

---

## 5. Projects — The Most Important Section

The **PROJECTS** section of the resume is already populated with the three worked
projects in `Mico_Data_Science_Projects/`, using **verified metrics** produced by
running each script end-to-end. You can either keep them as-is or swap in your own.

### The three provided projects (all runnable and verified)

| # | Project | Folder | Verified results |
|---|---------|--------|-------------------|
| 1 | E-Commerce Customer Purchase Behavior — EDA | `project_1_eda_customer_behavior` | ANOVA F = 18.70, p < 0.001; Age↔Purchase correlation 0.050 |
| 2 | House Price Prediction — Multiple Linear Regression | `project_2_regression_house_prices` | R² = 0.866, Adj. R² = 0.863, MAE ₹3.45L, RMSE ₹4.42L |
| 3 | Customer Churn Prediction — Logistic Regression + SMOTE | `project_3_classification_churn` | ROC-AUC 0.684; churn recall 0.68 vs 0.61 retained |

### If you substitute your own projects

Use the same action-oriented style: *Action verb + What you did + Tool used + Outcome/Insight.*

1. **Replace the title** with something specific.
2. **Keep the bullet points** — they mirror Mico's responsibilities (EDA, missing values,
   outliers, regression, hypothesis testing, feature engineering, documentation).
3. **Only quote a metric you can reproduce.** Every number in the resume above came from
   an actual script run with `random_state=42`; if you change the data, re-run and update.

### Running the projects yourself

```bash
cd Mico_Data_Science_Projects
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python generate_data.py         # creates all three datasets
python project_1_eda_customer_behavior/eda_analysis.py
python project_2_regression_house_prices/regression_model.py
python project_3_classification_churn/classification_model.py
```

Or launch the interactive dashboards:

```bash
streamlit run project_1_eda_customer_behavior/streamlit_app.py
streamlit run project_2_regression_house_prices/streamlit_app.py
streamlit run project_3_classification_churn/streamlit_app.py
```

> **Important:** `requirements.txt` uses **bounded ranges** (not exact pins) so pip can always
> find prebuilt wheels on Python 3.11–3.13. Do **not** let `pandas` reach 3.0 — seaborn 0.13 is
> broken against pandas ≥ 3.0 (`SeriesGroupBy.grouper` was removed), which crashes every
> `boxplot`/`catplot` call. The `pandas < 3.0` bound in `requirements.txt` enforces this.

Push the code (scripts + Streamlit apps) to GitHub and link the repo — and the live Streamlit demo URLs — from your resume.

**Live Streamlit dashboards (deployed):**

| Project | Live demo |
|---------|-----------|
| 1 — E-Commerce Customer Behavior (EDA) | https://mico-data-science-intern-kte8yj6a7aaryhfnj3oyab.streamlit.app/ |
| 2 — House Price Prediction (Regression) | https://mico-data-science-intern-uz5cau86ztv5kmua6r9jbv.streamlit.app/ |
| 3 — Customer Churn Prediction (Classification) | https://mico-data-science-intern-hcq56yhkqpkmgjaf9clrhz.streamlit.app/ |

Repository: https://github.com/Mohd-Abdul-Nawaz/mico-data-science-intern

---

## 6. Achievements & Positions of Responsibility

The resume includes one verified achievement:

- **150+ LeetCode problems solved** (data structures & algorithms) — evidence of consistent
  problem-solving practice relevant to a data science role.

Add further items only when they are truthful and defensible, for example:
- Secured X percentile in a hackathon or coding competition
- Organized a tech fest or club event
- Volunteer at a college event

> **Certifications were intentionally left out** of this resume — add a separate section back
> only if you earn one that is directly relevant to the role.

---

## 7. Declaration

Fill in Place and Date, then type your name at the bottom. When you convert to PDF for submission, you can either sign it or just type your name.

---

## 8. Final Steps Before Sending

1. **Save as PDF:** Convert the Markdown file to PDF (use any Markdown-to-PDF tool, or copy into Word/Google Docs and export as PDF). Name the file professionally:  
   `FirstName_LastName_Data_Science_Intern_Resume.pdf`

2. **ATS Check:** Make sure there are no tables, images, graphics, or columns. This template is plain text — perfect for ATS (Applicant Tracking Systems).

3. **Proofread:** Run a spellcheck. Ensure your email and phone number are correct.

4. **Tailor per application:** If applying to other companies, swap Mico-specific details (e.g., LINE/SMS/RCS mentions) for that company's product.

---

## 9. File Checklist

- [x] All `[BRACKETS]` replaced with real info
- [x] All three projects filled in with verified, reproducible metrics
- [ ] Only skills you can defend are listed
- [ ] Saved as PDF with professional filename
- [ ] GitHub repo and the three live Streamlit demo links are public and working

Good luck with your Mico application!
