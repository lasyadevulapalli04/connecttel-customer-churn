"""
Explainable AI (SHAP) Module for ConnectTel Customer Churn Prediction.

Extracts post-preprocessing feature names, computes SHAP values for feature importance,
generates summary plots, and provides individual customer-level churn risk explanations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """
    Extracts transformed feature names from fitted ColumnTransformer.
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted ColumnTransformer.
        
    Returns:
        list: List of output feature names after scaling and one-hot encoding.
    """
    feature_names = []
    
    try:
        # Scikit-learn >= 1.0 supports get_feature_names_out()
        feature_names = list(preprocessor.get_feature_names_out())
        # Clean prefix names (e.g. 'num__tenure' -> 'tenure', 'cat__Contract_One year' -> 'Contract_One year')
        cleaned_names = [name.split('__')[-1] for name in feature_names]
        return cleaned_names
    except Exception as e:
        print(f"[WARNING] Could not retrieve feature names automatically: {e}")
        return []


def generate_shap_explanation(model, X_preprocessed: np.ndarray, feature_names: list = None) -> tuple:
    """
    Computes SHAP values using TreeExplainer or LinearExplainer.
    
    Parameters:
        model: Trained classifier (or model inside pipeline).
        X_preprocessed (np.ndarray or DataFrame): Preprocessed input features.
        feature_names (list): Feature column names.
        
    Returns:
        tuple: (explainer, shap_values)
    """
    model_type = type(model).__name__
    print(f"[INFO] Computing SHAP values for model type: {model_type}...")
    
    if "XGB" in model_type or "RandomForest" in model_type or "Tree" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_preprocessed)
    else:
        # Linear models or fallback
        explainer = shap.LinearExplainer(model, X_preprocessed)
        shap_values = explainer.shap_values(X_preprocessed)
        
    # Handle list outputs for binary classification (take positive class churn=1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    return explainer, shap_values


def plot_shap_summary(shap_values, X_preprocessed, feature_names: list = None, save_path: str = None):
    """
    Plots and saves SHAP summary dot plot demonstrating feature impacts on churn predictions.
    
    Parameters:
        shap_values: Computed SHAP values.
        X_preprocessed: Preprocessed dataset matrix.
        feature_names (list): Feature names.
        save_path (str): File path to save figure.
    """
    plt.figure(figsize=(10, 6))
    
    if feature_names is not None:
        X_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        shap.summary_plot(shap_values, X_df, show=False)
    else:
        shap.summary_plot(shap_values, X_preprocessed, show=False)
        
    plt.title("SHAP Feature Importance & Impact Summary", fontsize=14, pad=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] SHAP summary plot saved to: {save_path}")
    plt.close()


def plot_shap_bar(shap_values, feature_names: list = None, save_path: str = None):
    """
    Plots and saves mean absolute SHAP feature importance bar chart.
    
    Parameters:
        shap_values: Computed SHAP values.
        feature_names (list): List of feature names.
        save_path (str): File path to save figure.
    """
    plt.figure(figsize=(10, 6))
    
    if feature_names is not None:
        X_df = pd.DataFrame(shap_values, columns=feature_names)
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, plot_type="bar", show=False)
        
    plt.title("Mean SHAP Feature Importance (|SHAP value|)", fontsize=14, pad=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] SHAP bar plot saved to: {save_path}")
    plt.close()
