# ConnectTel Project Completion & Manual Execution Checklist

Use this checklist to track your manual step-by-step execution, testing, and submission preparation for the ConnectTel Customer Churn Prediction project.

---

## Phase 1: Environment & Dataset Setup
- [x] Python virtual environment created and activated
- [x] Requirements installed via `pip install -r requirements.txt`
- [x] Telco Customer Churn dataset downloaded from Kaggle
- [x] Dataset placed at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Phase 2: Exploratory Data Analysis (EDA) & Feature Engineering
- [x] Notebook `01_Data_Loading_and_EDA.ipynb` executed manually
- [x] Data shape, missing values, and blank space strings verified
- [x] Target churn class distribution inspected (~26.54% churn)
- [x] Categorical feature charts vs churn generated
- [x] Notebook `02_Feature_Engineering_and_Preprocessing.ipynb` executed manually
- [x] Feature engineering verified (`TotalChargesPerTenure`, `ServiceCount`, `IsMonthlyContract`)
- [x] Preprocessing pipeline (`ColumnTransformer`) verified

---

## Phase 3: Model Training & Evaluation
- [x] Notebook `03_Model_Training_and_CrossValidation.ipynb` executed manually
- [x] Baseline models trained (Logistic Regression, Random Forest, XGBoost)
- [x] 5-Fold Stratified Cross-Validation metrics recorded
- [x] Controlled hyperparameter tuning (`GridSearchCV`) executed
- [x] Notebook `04_Evaluation_SHAP_and_Business_Insights.ipynb` executed manually
- [x] Model metrics evaluated on 20% test dataset (Accuracy: 80.62%, F1: 0.6061, ROC-AUC: 0.8422)
- [x] Confusion matrices and ROC curves generated and saved to `outputs/figures/`
- [x] Best model selected based on empirical metric results (Logistic Regression)
- [x] SHAP summary and feature importance plots generated
- [x] Final model (`models/best_model.joblib`) and preprocessor (`models/preprocessing_pipeline.joblib`) saved

---

## Phase 4: Web Application Testing & Deployment
- [x] Streamlit application launched locally via `streamlit run app/app.py`
- [x] Sidebar customer attribute inputs tested with sample values
- [x] Churn probability score, risk category badge, and retention tips verified
- [ ] Streamlit app deployed to cloud service (e.g., Streamlit Community Cloud)
- [ ] Real live application URL added to `submission/Live_Project_Link/Live_Project_Link.txt`

---

## Phase 5: Documentation & Submission Package Preparation
- [x] `docs/Project_Report.md` updated with actual empirical metric results
- [x] `docs/Presentation_Content.md` updated with actual metrics and slide notes
- [ ] Final Presentation PPT saved to `submission/PPT/`
- [ ] Final Report PDF saved to `submission/PDF/`
- [x] Source code copied to `submission/Source_Code/`
- [ ] Submission folder review completed
- [ ] Final submission ZIP archive manually created (`ConnectTel_Customer_Churn_Submission.zip`)
