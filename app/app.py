"""
ConnectTel Churn Predictor - Streamlit Web Application

Interactive machine-learning application for predicting telecom customer churn,
visualizing churn risk probabilities, and generating retention recommendations.
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

# Add project root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction import ChurnPredictor

# Page Configuration
st.set_page_config(
    page_title="ConnectTel Churn Predictor",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Header Section
    st.markdown('<p class="main-header">📶 ConnectTel Customer Churn Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn risk probabilities and receive actionable retention strategies.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load Predictor Model
    predictor = ChurnPredictor(
        model_path="models/best_model.joblib",
        pipeline_path="models/preprocessing_pipeline.joblib"
    )

    if not predictor.is_loaded:
        st.warning("⚠️ **Model Artifacts Not Detected**")
        st.info(
            "The trained model file (`models/best_model.joblib`) was not found.\n\n"
            "**Steps to activate the live model:**\n"
            "1. Open notebook `notebooks/03_Model_Training_and_CrossValidation.ipynb` or `notebooks/04_Evaluation_SHAP_and_Business_Insights.ipynb`.\n"
            "2. Run all code cells to train and save the model pipeline.\n"
            "3. Refresh this Streamlit app page once the model artifact is generated in the `models/` directory."
        )

    # Sidebar Form Input
    st.sidebar.header("📋 Customer Attribute Input")

    with st.sidebar:
        st.subheader("1. Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.subheader("2. Subscription & Services")
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        st.subheader("3. Add-on Services")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.subheader("4. Account & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=70.0, step=0.5)
        
        # Estimate total charges based on tenure if default
        est_total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(est_total_charges), step=10.0)

    # Prepare input dictionary
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Layout Main Panel into two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Customer Summary Profile")
        st.json(customer_data)

    with col2:
        st.subheader("📊 Churn Risk Prediction")
        
        if st.button("Predict Customer Churn Risk"):
            if predictor.is_loaded:
                result = predictor.predict_single_customer(customer_data)
                
                if result['status'] == 'success':
                    prob = result['churn_probability']
                    pct = result['churn_probability_pct']
                    tier = result['risk_tier']
                    color = result['risk_color']
                    label = result['prediction_label']

                    # Probability Metric
                    st.metric(
                        label="Predicted Churn Probability",
                        value=f"{pct}%",
                        delta=f"{tier}",
                        delta_color="inverse" if color == "red" else "normal"
                    )

                    # Visual Progress Bar
                    st.progress(prob)

                    # Risk Status Callout Box
                    if color == "red":
                        st.error(f"🚨 **HIGH RISK**: The customer has a **{pct}%** chance of churning. Immediate retention intervention recommended.")
                    elif color == "orange":
                        st.warning(f"⚠️ **MEDIUM RISK**: The customer has a **{pct}%** chance of churning. Consider offering proactive contract extensions or add-on discounts.")
                    else:
                        st.success(f"✅ **LOW RISK**: The customer has a **{pct}%** chance of churning. High likelihood of retention.")

                    # Recommended Retention Action Plan
                    st.subheader("💡 Recommended Retention Actions")
                    if contract == "Month-to-month":
                        st.write("• **Contract Incentive**: Offer a 10% discount on a 1-Year or 2-Year contract commitment.")
                    if tech_support == "No" and internet_service != "No":
                        st.write("• **Tech Support Add-on**: Provide 3 months free Tech Support to improve service stability.")
                    if payment_method == "Electronic check":
                        st.write("• **Billing Upgrade**: Promote automatic credit card or bank transfer payment methods to avoid billing friction.")
                    if tenure <= 12:
                        st.write("• **Onboarding Support**: Conduct a new customer satisfaction check-in during the first year.")

                else:
                    st.error(result['message'])
            else:
                st.error("Cannot perform prediction because trained model file was not found in `models/best_model.joblib`.")

if __name__ == "__main__":
    main()
