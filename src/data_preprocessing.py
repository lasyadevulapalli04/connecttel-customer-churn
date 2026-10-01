"""
Data Preprocessing Module for ConnectTel Customer Churn Prediction.

Handles loading raw data, cleaning data quality issues (missing total charges,
data types, spaces), target encoding, and building scikit-learn preprocessing pipelines.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw Telco Customer Churn CSV dataset.
    
    Parameters:
        file_path (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Raw dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. "
            "Please ensure the CSV file is placed in the 'data/' directory."
        )
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Telco Customer Churn data.
    - Handles blank string spaces in TotalCharges (converts to float, fills NaNs).
    - Encodes Churn target column ('Yes' -> 1, 'No' -> 0).
    - Removes customerID column if present.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_clean = df.copy()
    
    # Strip leading/trailing whitespaces from object columns
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Handle TotalCharges blank strings (' ')
    if 'TotalCharges' in df_clean.columns:
        # Replace blank strings or invalid values with NaN and convert to float
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'].replace(['', ' '], np.nan),
            errors='coerce'
        )
        
        # For new customers with tenure = 0, TotalCharges is typically 0.0
        missing_count = df_clean['TotalCharges'].isnull().sum()
        if missing_count > 0:
            print(f"[INFO] Found {missing_count} missing values in TotalCharges. Imputing with 0.0.")
            df_clean['TotalCharges'].fillna(0.0, inplace=True)
            
    # Convert Churn target variable to binary integer (1 = Churned, 0 = Retained)
    if 'Churn' in df_clean.columns:
        if df_clean['Churn'].dtype == object or isinstance(df_clean['Churn'].iloc[0], str):
            churn_mapping = {'Yes': 1, 'No': 0, '1': 1, '0': 0}
            df_clean['Churn'] = df_clean['Churn'].map(churn_mapping).astype(int)
            print("[INFO] Target column 'Churn' mapped to binary (Yes: 1, No: 0).")
            
    print(f"[INFO] Cleaning complete. Cleaned shape: {df_clean.shape}")
    return df_clean


def get_feature_columns(df: pd.DataFrame, target_col: str = 'Churn', id_col: str = 'customerID'):
    """
    Identifies numerical and categorical feature columns for preprocessing.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        target_col (str): Target column name.
        id_col (str): Unique customer ID column name to exclude.
        
    Returns:
        tuple: (numerical_cols, categorical_cols)
    """
    features_df = df.drop(columns=[col for col in [target_col, id_col] if col in df.columns])
    
    # Identify numerical columns (include SeniorCitizen as numerical or categorical based on preference)
    numerical_cols = features_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numerical_cols, categorical_cols


def build_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for preprocessing.
    - Numerical: SimpleImputer (median) -> StandardScaler
    - Categorical: OneHotEncoder (drop='first', handle_unknown='ignore')
    
    Parameters:
        numerical_cols (list): List of numeric column names.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        ColumnTransformer: Scikit-learn ColumnTransformer instance.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
