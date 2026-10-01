# Models Directory

This directory stores saved serialized artifacts generated after model training, evaluation, and selection.

## Expected Artifacts

After you manually execute `notebooks/04_Evaluation_SHAP_and_Business_Insights.ipynb` or run the training scripts, the following joblib files will be saved here:

1. `best_model.joblib`: The trained scikit-learn or XGBoost classification model selected based on empirical evaluation.
2. `preprocessing_pipeline.joblib`: The fitted scikit-learn `ColumnTransformer` / `Pipeline` used to preprocess numerical and categorical features.
3. `feature_names.joblib`: (Optional) Saved list of post-encoding feature names for SHAP and model explainability.

## Notes

- Do not commit large model binary files to version control if they exceed storage limits.
- The Streamlit web application (`app/app.py`) loads these exact filenames to serve real-time predictions.
