"""
Evaluation Module for ConnectTel Customer Churn Prediction.

Calculates classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC),
plots Confusion Matrices and ROC Curves, and formats comparative model evaluation tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> dict:
    """
    Evaluates a trained model pipeline on test dataset.
    
    Parameters:
        model: Trained scikit-learn model or Pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test target labels.
        model_name (str): Display name for the model.
        
    Returns:
        dict: Metric scores dictionary.
    """
    y_pred = model.predict(X_test)
    
    # Predict probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'Confusion Matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"\n==================== {model_name} Evaluation ====================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("==================================================================")
    
    return results


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model", save_path: str = None):
    """
    Plots and saves confusion matrix heatmap.
    
    Parameters:
        cm (np.ndarray): 2x2 confusion matrix array.
        model_name (str): Model name for title.
        save_path (str): File path to save figure.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Confusion matrix figure saved to: {save_path}")
    plt.close()


def plot_roc_curves(eval_results_list: list, y_test: pd.Series, save_path: str = None):
    """
    Overlays ROC curves for multiple evaluated models on a single figure.
    
    Parameters:
        eval_results_list (list): List of evaluation result dictionaries from evaluate_model().
        y_test (pd.Series): True test target labels.
        save_path (str): File path to save figure.
    """
    plt.figure(figsize=(8, 6))
    
    for res in eval_results_list:
        model_name = res['Model']
        y_prob = res['y_prob']
        auc = res['ROC-AUC']
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, pad=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] ROC Curves figure saved to: {save_path}")
    plt.close()


def create_model_comparison_table(eval_results_list: list) -> pd.DataFrame:
    """
    Creates a formatted comparison table DataFrame for model metric evaluation.
    
    Parameters:
        eval_results_list (list): List of evaluation result dicts.
        
    Returns:
        pd.DataFrame: Metric comparison table sorted by F1-Score.
    """
    data = []
    for res in eval_results_list:
        data.append({
            'Model': res['Model'],
            'Accuracy': res['Accuracy'],
            'Precision': res['Precision'],
            'Recall': res['Recall'],
            'F1-Score': res['F1-Score'],
            'ROC-AUC': res['ROC-AUC']
        })
        
    df_comp = pd.DataFrame(data)
    df_comp = df_comp.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    return df_comp
