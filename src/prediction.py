"""
Prediction Module for ConnectTel Customer Churn Prediction.

Loads saved trained model pipelines and provides single-customer or batch inference functions.
Returns predicted churn label, probability, risk tier, and key risk drivers.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import create_features


class ChurnPredictor:
    """
    Inference wrapper class forConnectTel Churn Predictor model.
    """
    
    def __init__(self, model_path: str = "models/best_model.joblib", pipeline_path: str = "models/preprocessing_pipeline.joblib"):
        """
        Initializes predictor by loading trained model artifacts.
        
        Parameters:
            model_path (str): Path to saved model joblib file.
            pipeline_path (str): Path to saved preprocessing pipeline joblib file.
        """
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        self.load_artifacts()
        
    def load_artifacts(self):
        """Loads model and preprocessing pipeline from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Loaded trained model from: {self.model_path}")
        else:
            print(f"[WARNING] Model file not found at: {self.model_path}")
            
        if os.path.exists(self.pipeline_path):
            self.pipeline = joblib.load(self.pipeline_path)
            print(f"[INFO] Loaded preprocessing pipeline from: {self.pipeline_path}")
        else:
            print(f"[WARNING] Preprocessing pipeline not found at: {self.pipeline_path}")
            
        if self.model is not None:
            self.is_loaded = True
            
    def predict_single_customer(self, customer_dict: dict) -> dict:
        """
        Predicts churn risk for a single customer input dictionary.
        
        Parameters:
            customer_dict (dict): Input key-value pairs matching dataset columns.
            
        Returns:
            dict: Structured prediction output.
        """
        if not self.is_loaded:
            return {
                'status': 'error',
                'message': 'Model artifacts are not loaded. Please train and save the model first.'
            }
            
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([customer_dict])
        
        # Apply feature engineering logic
        input_df_feat = create_features(input_df)
        
        # Predict probability and class
        try:
            # If pipeline is embedded in full model pipeline or separate
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(input_df_feat)[0, 1]
                pred_class = int(self.model.predict(input_df_feat)[0])
            else:
                prob = float(self.model.predict(input_df_feat)[0])
                pred_class = 1 if prob >= 0.5 else 0
                
            # Determine Risk Category Tier
            if prob < 0.35:
                risk_tier = "Low Churn Risk"
                color = "green"
            elif prob < 0.65:
                risk_tier = "Medium Churn Risk"
                color = "orange"
            else:
                risk_tier = "High Churn Risk"
                color = "red"
                
            prediction_label = "Churn (Yes)" if pred_class == 1 else "Retained (No)"
            
            return {
                'status': 'success',
                'prediction_class': pred_class,
                'prediction_label': prediction_label,
                'churn_probability': prob,
                'churn_probability_pct': round(prob * 100, 2),
                'risk_tier': risk_tier,
                'risk_color': color
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Inference failed with error: {str(e)}"
            }
