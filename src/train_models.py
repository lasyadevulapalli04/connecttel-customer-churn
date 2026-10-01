"""
Model Training Module for ConnectTel Customer Churn Prediction.

Defines model architectures (Logistic Regression, Random Forest, XGBoost),
executes Stratified K-Fold Cross-Validation, and performs controlled hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline


def get_baseline_models(random_state: int = 42) -> dict:
    """
    Initializes candidate classification models with reasonable default parameters.
    
    Models:
    1. Logistic Regression: Interpretable baseline classifier.
    2. Random Forest: Tree-based ensemble capable of non-linear patterns.
    3. XGBoost: Advanced gradient boosting classifier.
    
    Parameters:
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Mapping of model names to model instances.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_cv_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series, preprocessor, cv_splits: int = 5) -> pd.DataFrame:
    """
    Evaluates candidate models using Stratified K-Fold Cross Validation.
    Combines preprocessing and modeling in a scikit-learn Pipeline to prevent data leakage.
    
    Parameters:
        models (dict): Dictionary of model instances.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        preprocessor (ColumnTransformer): Fitted/unfitted ColumnTransformer preprocessor.
        cv_splits (int): Number of cross-validation folds.
        
    Returns:
        pd.DataFrame: Summary table of cross-validation metrics (mean +/- std).
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = []
    
    for name, model in models.items():
        print(f"[INFO] Running {cv_splits}-Fold CV for model: {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        scores = cross_validate(
            pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )
        
        cv_results.append({
            'Model': name,
            'Accuracy': f"{np.mean(scores['test_accuracy']):.4f} (+/- {np.std(scores['test_accuracy']):.4f})",
            'Precision': f"{np.mean(scores['test_precision']):.4f} (+/- {np.std(scores['test_precision']):.4f})",
            'Recall': f"{np.mean(scores['test_recall']):.4f} (+/- {np.std(scores['test_recall']):.4f})",
            'F1-Score': f"{np.mean(scores['test_f1']):.4f} (+/- {np.std(scores['test_f1']):.4f})",
            'ROC-AUC': f"{np.mean(scores['test_roc_auc']):.4f} (+/- {np.std(scores['test_roc_auc']):.4f})"
        })
        
    return pd.DataFrame(cv_results)


def tune_hyperparameters(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> tuple:
    """
    Performs controlled, lightweight GridSearchCV hyperparameter tuning for a selected model type.
    
    Parameters:
        model_type (str): Name of model to tune ('Random Forest' or 'XGBoost' or 'Logistic Regression').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        preprocessor: Preprocessing pipeline.
        
    Returns:
        tuple: (best_estimator, best_params, best_score)
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    if model_type == 'Logistic Regression':
        base_model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    elif model_type == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5]
        }
    elif model_type == 'XGBoost':
        base_model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])
    
    print(f"[INFO] Starting GridSearchCV tuning for {model_type}...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"[INFO] Best Parameters for {model_type}: {grid_search.best_params_}")
    print(f"[INFO] Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
