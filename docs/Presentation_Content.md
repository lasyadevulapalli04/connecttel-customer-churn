# ConnectTel Customer Churn Prediction
## Presentation Slide Content Outline (12 Slides)

---

### Slide 1: Title Slide
- **Title:** Predictive Customer Churn Model for ConnectTel
- **Subtitle:** An End-to-End Machine Learning Solution for Telecom Retention
- **Presenter:** [INSERT YOUR NAME]
- **Date:** October 1, 2026

---

### Slide 2: Executive Summary & Business Problem
- **Context:** ConnectTel seeks to reduce customer attrition in a competitive telecom market.
- **Problem Statement:** High churn reduces revenue and increases customer acquisition costs (CAC).
- **Goal:** Predict churn probabilities, understand key drivers, and enable targeted retention strategies.

---

### Slide 3: Dataset Overview & Key Attributes
- **Source:** Telco Customer Churn Dataset (7,043 customer records).
- **Features:** Demographics, Account Tenure, Monthly & Total Charges, Subscribed Services, Contract Type.
- **Target Variable:** `Churn` (Binary: Yes = 1, No = 0; 26.54% positive churn baseline).

---

### Slide 4: Data Preprocessing & Pipeline Architecture
- **Cleaning:** Handled 11 blank space strings in `TotalCharges` (`tenure = 0`).
- **Engineered Features:**
  - `TotalChargesPerTenure`
  - `ServiceCount`
  - `IsMonthlyContract`
  - `HasTechSupport`
- **Pipeline:** Integrated `StandardScaler` and `OneHotEncoder` via `ColumnTransformer`.

---

### Slide 5: Exploratory Data Analysis (EDA) Highlights
- **Target Ratio:** 26.54% Churn vs 73.46% Retained.
- **Key Insight 1:** Month-to-month contract holders churn at ~42.7% vs ~2.8% for 2-Year contracts.
- **Key Insight 2:** Fiber Optic users churn at ~41.9% due to higher monthly bill rates ($70–$110).
- **Key Insight 3:** Customers with add-on services (Tech Support, Online Security) stay longer.

---

### Slide 6: Machine Learning Models & Cross-Validation Strategy
- **Models Evaluated:**
  1. Logistic Regression (Baseline Classifier)
  2. Random Forest Classifier (Tree Ensemble)
  3. XGBoost Classifier (Gradient Boosting)
- **Validation:** 5-Fold Stratified Cross-Validation inside scikit-learn pipelines.

---

### Slide 7: Model Performance & Comparison

#### Test Set Evaluation Metrics (1,409 Holdout Customer Records)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| 🏆 **Logistic Regression** | **0.8062** | **0.6583** | **0.5615** | **0.6061** | **0.8422** |
| **Random Forest** | 0.7977 | 0.6469 | 0.5241 | 0.5790 | **0.8428** |
| **XGBoost** | 0.7935 | 0.6334 | 0.5267 | 0.5752 | 0.8409 |

- **Winner Selection:** **Logistic Regression** achieved the highest test **F1-Score (0.6061)** and highest **Recall (0.5615)**.

---

### Slide 8: Explainable AI with SHAP
- **Objective:** Demystify black-box machine learning predictions.
- **Top Risk Drivers identified by SHAP:**
  - `Contract_Month-to-month` (+0.85 SHAP impact on churn probability)
  - `tenure` (-0.72 SHAP impact on churn probability)
  - `InternetService_Fiber optic` (+0.54 SHAP impact on churn probability)
- **Key finding:** Contract commitment and customer tenure are primary retention drivers.

---

### Slide 9: ConnectTel Streamlit Web Application
- **Live Demo Overview:**
  - Interactive customer attribute inputs in sidebar.
  - Real-time feature calculation and model inference using `models/best_model.joblib`.
  - Risk classification badge (Low / Medium / High Risk).
  - Actionable retention recommendation suggestions.

---

### Slide 10: Business Impact & Strategic Recommendations
- **Recommendation 1:** Offer 15% contract migration discounts for Month-to-month subscribers.
- **Recommendation 2:** Bundle Tech Support add-ons for Fiber Optic internet subscribers.
- **Recommendation 3:** Promote automated billing channels (Direct Debit / Credit Card).

---

### Slide 11: Limitations & Future Scope
- **Limitations:** Absence of real-time call center log data and network quality metrics.
- **Future Enhancements:** Integrate survival analysis and real-time API scoring endpoints.

---

### Slide 12: Q&A / Conclusion
- Thank you for your time.
- Open for questions and viva evaluation.
