# ConnectTel Telecom Customer Churn Prediction
## Comprehensive Project Report

**Author:** [INSERT YOUR NAME]  
**Case Study:** ConnectTel  
**Domain:** Machine Learning  
**Submission Date:** October 1, 2026  

---

## 1. Abstract

Customer churn poses a significant challenge in the telecommunications industry, directly impacting revenue retention and customer lifetime value. This project develops an end-to-end, reproducible predictive machine learning system for **ConnectTel** to identify customers at high risk of churning before cancellation occurs. Using the standard Telco Customer Churn dataset (7,043 customer records), we perform data cleaning, exploratory data analysis (EDA), domain feature engineering, and train candidate classifiers including Logistic Regression, Random Forest, and XGBoost inside scikit-learn pipelines with 5-Fold Stratified Cross-Validation. Explainable AI techniques using SHAP interpret model risk drivers, and a Streamlit web interface is created to serve real-time predictions.

---

## 2. Introduction & Business Problem

ConnectTel is a telecommunications provider seeking to reduce customer churn rates. Acquiring a new subscriber is estimated to cost 5 to 25 times more than retaining an existing subscriber. By accurately identifying churn risk factors, ConnectTel's retention team can proactively target vulnerable customer segments with tailored loyalty programs and contract incentives.

### Project Objectives
1. Perform thorough EDA to uncover key behavioral and demographic churn drivers.
2. Build an automated preprocessing and feature engineering pipeline to transform raw data cleanly without data leakage.
3. Compare Logistic Regression, Random Forest, and XGBoost models using Stratified K-Fold Cross Validation.
4. Evaluate candidate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
5. Apply SHAP explainability to quantify feature contributions to customer churn risk.
6. Build a Streamlit web application (`ConnectTel Churn Predictor`) to serve customer predictions interactively.

---

## 3. Dataset Description

The project utilizes the standard publicly available Telco Customer Churn dataset (7,043 rows x 21 columns).

### Feature Taxonomy
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Services:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Subscribed Products:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable:** `Churn` (Yes = 1, No = 0; ~26.5% baseline churn rate)

---

## 4. Data Preprocessing & Feature Engineering

### Data Cleaning
- **TotalCharges Imputation:** Converted blank space strings `' '` to `NaN` (11 rows where `tenure = 0`) and imputed with `0.0`.
- **Target Mapping:** Converted string target `'Yes'`/`'No'` to binary integers `1`/`0`.

### Engineered Features
1. `TotalChargesPerTenure`: Ratio of `TotalCharges / tenure` (or `MonthlyCharges` if `tenure = 0`).
2. `ServiceCount`: Total count of active subscribed value-added services.
3. `IsMonthlyContract`: Binary flag for high-risk Month-to-month contracts.
4. `HasTechSupport`: Binary flag indicating presence of dedicated technical support.

### Scikit-Learn Pipeline
- **Numerical Features:** `SimpleImputer(strategy='median')` -> `StandardScaler()`
- **Categorical Features:** `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(drop='first', handle_unknown='ignore')`

---

## 5. Model Development & Evaluation Results

### 5-Fold Stratified Cross-Validation Metrics

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-Score | CV ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.8021** (±0.0116) | 0.6521 (±0.0248) | **0.5452** (±0.0435) | **0.5931** (±0.0305) | **0.8456** (±0.0125) |
| **Random Forest** | 0.8007 (±0.0137) | **0.6628** (±0.0379) | 0.5090 (±0.0270) | 0.5754 (±0.0275) | 0.8428 (±0.0109) |
| **XGBoost** | 0.7980 (±0.0111) | 0.6496 (±0.0281) | 0.5197 (±0.0276) | 0.5771 (±0.0244) | 0.8435 (±0.0100) |

### Test Set Evaluation Metrics (20% Holdout Split - 1,409 Records)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|
| 🏆 **Logistic Regression** | **0.8062** (80.62%) | **0.6583** (65.83%) | **0.5615** (56.15%) | **0.6061** (60.61%) | **0.8422** (84.22%) |
| **Random Forest** | 0.7977 (79.77%) | 0.6469 (64.69%) | 0.5241 (52.41%) | 0.5790 (57.90%) | **0.8428** (84.28%) |
| **XGBoost** | 0.7935 (79.35%) | 0.6334 (63.34%) | 0.5267 (52.67%) | 0.5752 (57.52%) | 0.8409 (84.09%) |

### Selected Best Model
**Logistic Regression** was selected as the final production model based on achieving the highest test **F1-Score (0.6061)** and highest **Recall (0.5615)**, capturing the highest proportion of churners while preserving strong interpretability.

---

## 6. SHAP Explainability & Key Drivers

SHAP (SHapley Additive exPlanations) analysis revealed the top feature contributions influencing churn:

1. **`Contract_Month-to-month`:** Positively associated with elevated churn risk probability.
2. **`tenure`:** Longer customer tenure significantly reduces churn likelihood.
3. **`InternetService_Fiber optic`:** Higher monthly fiber optic charges increase churn probability.
4. **`TechSupport_No` & `OnlineSecurity_No`:** Absence of support and security add-ons correlates with higher churn risk.

---

## 7. Business Recommendations

Based on empirical data findings:
1. **Contract Strategy:** Offer 15% 1-Year contract migration discounts to Month-to-month customers.
2. **Technical Support Promotion:** Bundle 3 months of free Tech Support for Fiber Optic subscribers during onboarding.
3. **Automatic Payment Discounts:** Encourage switching from Electronic Check to direct debit or credit card payment methods.

---

## 8. Conclusion & Future Improvements

### Conclusion
The ConnectTel Churn Predictor provides an end-to-end framework for identifying at-risk subscribers and deploying targeted retention workflows.

### Future Scope
- Incorporate customer network usage metrics (data consumption, dropped call counts).
- Implement time-series survival analysis to model customer lifetime duration.
