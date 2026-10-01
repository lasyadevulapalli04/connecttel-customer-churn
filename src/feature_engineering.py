"""
Feature Engineering Module for ConnectTel Customer Churn Prediction.

Creates domain-specific derived features to improve machine-learning model performance,
such as TotalChargesPerTenure, ServiceCount, and Contract indicators.
"""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new business domain features from raw attributes.
    
    Derived Features:
    1. TotalChargesPerTenure: Average total charges per tenure month (handles 0 tenure safely).
    2. ServiceCount: Total number of active subscribed value-added services.
    3. IsMonthlyContract: Binary flag indicating high-risk Month-to-month contracts.
    4. HasTechSupport: Binary flag indicating presence of tech support.
    
    Parameters:
        df (pd.DataFrame): Cleaned input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe augmented with engineered features.
    """
    df_feat = df.copy()
    
    # 1. TotalChargesPerTenure
    # Ratio of TotalCharges to tenure. If tenure is 0, fall back to MonthlyCharges.
    if 'TotalCharges' in df_feat.columns and 'tenure' in df_feat.columns:
        df_feat['TotalChargesPerTenure'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['TotalCharges'] / df_feat['tenure'],
            df_feat['MonthlyCharges']
        )
        print("[INFO] Feature engineered: 'TotalChargesPerTenure'")
        
    # 2. ServiceCount
    # Count of active subscribed telecom services
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    present_services = [col for col in service_cols if col in df_feat.columns]
    
    if present_services:
        # Count how many of these columns equal 'Yes'
        df_feat['ServiceCount'] = (df_feat[present_services] == 'Yes').sum(axis=1)
        print("[INFO] Feature engineered: 'ServiceCount'")
        
    # 3. IsMonthlyContract
    if 'Contract' in df_feat.columns:
        df_feat['IsMonthlyContract'] = np.where(df_feat['Contract'] == 'Month-to-month', 1, 0)
        print("[INFO] Feature engineered: 'IsMonthlyContract'")
        
    # 4. HasTechSupport
    if 'TechSupport' in df_feat.columns:
        df_feat['HasTechSupport'] = np.where(df_feat['TechSupport'] == 'Yes', 1, 0)
        print("[INFO] Feature engineered: 'HasTechSupport'")
        
    return df_feat
