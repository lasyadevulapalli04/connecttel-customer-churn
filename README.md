# ConnectTel Customer Churn Prediction Model

**Project Title:** Predictive Customer Churn Model for a Telecom Company  
**Case Study:** ConnectTel  
**Domain:** Machine Learning  
**Deadline:** October 1, 2026  
**Project Mode:** Internship Portfolio Project (Manual Execution Mode)  

---

## 1. Project Overview
This project delivers a complete end-to-end machine learning solution to predict customer churn for **ConnectTel**, a telecommunications provider. By accurately identifying subscribers who are likely to cancel their subscriptions, ConnectTel can deploy targeted customer retention campaigns, reduce customer acquisition costs (CAC), and preserve monthly recurring revenue (MRR).

---

## 2. Business Problem
Customer retention is crucial in the telecom industry where subscriber acquisition cost is high. ConnectTel faces customer attrition due to pricing friction, contract terms, service quality, and market competition. A machine-learning system is needed to score every customer's churn risk probability and explain the primary drivers behind their risk level.

---

## 3. Project Objectives
1. Perform comprehensive Exploratory Data Analysis (EDA) on telecom customer demographic and service usage data.
2. Build an automated scikit-learn preprocessing pipeline to scale numeric features and one-hot encode categorical features without data leakage.
3. Engineer domain-specific features (`TotalChargesPerTenure`, `ServiceCount`, `IsMonthlyContract`, `HasTechSupport`).
4. Train and compare baseline machine learning models: **Logistic Regression**, **Random Forest**, and **XGBoost**.
5. Perform 5-Fold Stratified Cross-Validation and controlled hyperparameter tuning with `GridSearchCV`.
6. Evaluate models on a 20% holdout test dataset using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
7. Apply **SHAP (SHapley Additive exPlanations)** to interpret model predictions and identify top churn risk factors.
8. Develop an interactive **Streamlit web application (`ConnectTel Churn Predictor`)** to serve real-time predictions and recommended retention strategies.

---

## 4. Dataset Information
- **Source:** Kaggle Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Records:** 7,043 rows x 21 columns
- **Target Variable:** `Churn` (Yes = 1, No = 0; ~26.5% positive churn rate)
- **Features Include:**
  - *Demographics:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - *Account Status:* `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - *Subscribed Services:* `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## 5. Technologies & Libraries
- **Language:** Python 3.9+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Model Explainability:** `shap`
- **Web Application:** `streamlit`
- **Model Serialization:** `joblib`
- **Environment:** `jupyter`

---

## 6. Machine Learning Workflow
```text
Data Loading & Cleaning ──► Feature Engineering ──► Preprocessing Pipeline ──► Train/Test Split
                                                                                     │
                                                                                     ▼
Streamlit Inference ◄── Model Artifact Saving ◄── SHAP Explainability ◄── CV & Model Evaluation
```

---

## 7. Installation & Environment Setup

### Step 1: Clone or Navigate to Project Directory
```bash
cd C:\Users\lasya\.gemini\antigravity-ide\scratch\Customer_Churn_Prediction
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Dataset Placement
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in the `data/` folder:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## 9. How to Execute the Project Step-by-Step

Execute the Jupyter Notebooks manually in sequential order:

1. **`notebooks/01_Data_Loading_and_EDA.ipynb`**
   - Loads raw data, cleans space strings in `TotalCharges`, encodes `Churn`, and generates EDA charts.
2. **`notebooks/02_Feature_Engineering_and_Preprocessing.ipynb`**
   - Engineers derived features (`TotalChargesPerTenure`, `ServiceCount`) and tests scikit-learn `ColumnTransformer`.
3. **`notebooks/03_Model_Training_and_CrossValidation.ipynb`**
   - Executes 5-Fold Stratified Cross-Validation on Logistic Regression, Random Forest, and XGBoost.
4. **`notebooks/04_Evaluation_SHAP_and_Business_Insights.ipynb`**
   - Evaluates models on test set, plots ROC curves and confusion matrices, computes SHAP values, and saves `models/best_model.joblib`.

---

## 10. How to Run the Streamlit Application
After generating the trained model artifact, launch the Streamlit app locally:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Project Structure
```text
Customer_Churn_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_Data_Loading_and_EDA.ipynb
│   ├── 02_Feature_Engineering_and_Preprocessing.ipynb
│   ├── 03_Model_Training_and_CrossValidation.ipynb
│   └── 04_Evaluation_SHAP_and_Business_Insights.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── prediction.py
├── app/
│   └── app.py
├── models/
│   └── README.md
├── outputs/
│   ├── figures/
│   ├── results/
│   └── demo/
├── docs/
│   ├── Project_Report.md
│   ├── Presentation_Content.md
│   ├── Business_Insights.md
│   └── PROJECT_COMPLETION_CHECKLIST.md
└── submission/
    ├── PPT/
    ├── PDF/
    ├── Source_Code/
    ├── Live_Project_Link/
    │   └── Live_Project_Link.txt
    └── README.txt
```

---

## 12. Limitations & Future Scope
- **Limitations:** Data is cross-sectional; real-world churn benefits from dynamic time-series log tracking.
- **Future Scope:** Incorporate customer service call interaction logs and customer lifetime value (CLV) estimation.

---

## 13. Internship Project Verification Note
> *Project structure and implementation created. Full execution has intentionally NOT been performed because the user will execute and verify the project manually.*
