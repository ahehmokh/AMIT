import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- Data Preprocessing ---
print("--- Data Preprocessing ---")

# Make a copy to avoid Editing the original DataFrame "CSV file"
df_processed = df.copy()

# Impute zeros with NaN for relevant columns before further processing
zero_to_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_to_nan_cols:
    df_processed[col] = df_processed[col].replace(0, np.nan)

# Use SimpleImputer for mean imputation for NaNs
imputer = SimpleImputer(strategy='mean')
df_processed[zero_to_nan_cols] = imputer.fit_transform(df_processed[zero_to_nan_cols])

# Outlier handling (using IQR method for example)
# Applying Winsorization (clipping) for outlier handling
numerical_cols = df_processed.select_dtypes(include=np.number).columns.drop('Outcome', errors='ignore')
for col in numerical_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
    

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

# BMI Category
df_processed['BMI_Category'] = pd.cut(df_processed['BMI'],
                                      bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III'],
                                      right=False)

# Glucose to Insulin Ratio (might indicate insulin resistance)
# Add a small epsilon to prevent division by zero
df_processed['Glucose_Insulin_Ratio'] = df_processed['Glucose'] / (df_processed['Insulin'].replace(0, 1e-6))

# BloodPressure_Age_Interaction
df_processed['BloodPressure_Age_Interaction'] = df_processed['BloodPressure'] * df_processed['Age']

# Has_Insulin (binary feature for insulin presence)
df_processed['Has_Insulin'] = (df_processed['Insulin'] > 0).astype(int)

# Pregnancy_Glucose_Interaction
df_processed['Pregnancy_Glucose_Interaction'] = df_processed['Pregnancies'] * df_processed['Glucose']

# Age_BMI_Interaction
df_processed['Age_BMI_Interaction'] = df_processed['Age'] * df_processed['BMI']

# One-hot encode BMI_Category
df_processed = pd.get_dummies(df_processed, columns=['BMI_Category'], prefix='BMI_Cat')

# --- Feature and target variable ---
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

# Ensure all columns are numeric after one-hot encoding
# Convert object columns to numeric, coercing errors will turn non-convertible values into NaN
X = X.apply(pd.to_numeric, errors='coerce')

# After converting to numeric, if any NaNs were introduced (e.g., from new categories in test data not seen in train),
# impute them. This step is crucial for consistency.
# Use a new imputer instance to avoid fitting on potentially already processed data in the pipeline.
nan_imputer_after_dummies = SimpleImputer(strategy='mean')
X = pd.DataFrame(nan_imputer_after_dummies.fit_transform(X), columns=X.columns)


# --- FIX: Save the exact feature columns for consistency with GUI ---
joblib.dump(X.columns.tolist(), 'model_feature_columns.pkl')
print(f"Saved model_feature_columns.pkl with {len(X.columns)} features.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model Training and Hyperparameter Optimization with RandomizedSearchCV ---
print("\n--- Model Training & Hyperparameter Optimization (RandomizedSearchCV) ---")

# Define pipelines with StandardScaler
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Parameter distributions for RandomizedSearchCV
param_dist_rf = {
    'classifier__n_estimators': [100, 150, 200, 250, 300, 400, 500],
    'classifier__max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0],
    'classifier__max_depth': [5, 10, 15, 20, 25, None],
    'classifier__min_samples_split': [2, 5, 10, 15, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 8, 10],
    'classifier__bootstrap': [True, False],
}

param_dist_gb = {
    'classifier__n_estimators': [100, 150, 200, 250, 300, 400, 500],
    'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_samples_split': [2, 5, 10, 15],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV for RandomForest
print("Tuning RandomForestClassifier with RandomizedSearchCV...")
random_search_rf = RandomizedSearchCV(pipeline_rf, param_distributions=param_dist_rf,
                                      n_iter=100,  # Increased iterations for a more thorough search
                                      cv=5, n_jobs=-1, verbose=1, scoring='roc_auc',
                                      random_state=42)
random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_
print(f"Best RandomForest Parameters: {random_search_rf.best_params_}")
print(f"Best RandomForest Cross-Validation AUC: {random_search_rf.best_score_:.4f}")

# RandomizedSearchCV for GradientBoosting
print("\nTuning GradientBoostingClassifier with RandomizedSearchCV...")
random_search_gb = RandomizedSearchCV(pipeline_gb, param_distributions=param_dist_gb,
                                      n_iter=100,  # Increased iterations
                                      cv=5, n_jobs=-1, verbose=1, scoring='roc_auc',
                                      random_state=42)
random_search_gb.fit(X_train, y_train)
best_gb_model = random_search_gb.best_estimator_
print(f"Best GradientBoosting Parameters: {random_search_gb.best_params_}")
print(f"Best GradientBoosting Cross-Validation AUC: {random_search_gb.best_score_:.4f}")

# --- Ensemble Method (VotingClassifier) ---
print("\n--- Ensemble Method (VotingClassifier) ---")

# Extract the best classifiers from their pipelines
# This assumes that the 'scaler' step in the individual best_models has been fitted correctly
# when those models were trained.
clf1 = best_rf_model.named_steps['classifier']
clf2 = best_gb_model.named_steps['classifier']

# Create a new pipeline for the ensemble, including a scaler.
# It's crucial that if individual models used scaling, the ensemble also applies it consistently.
# The scaler here will be fitted on X_train when ensemble_pipeline.fit(X_train, y_train) is called.
ensemble_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('voting_classifier', VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2)],
        voting='soft',  # 'soft' voting for probability averaging
        weights=[0.5, 0.5] # Equal weights, can be optimized further if needed
    ))
])

# Fit the ensemble pipeline
ensemble_pipeline.fit(X_train, y_train)

# --- Evaluation ---
print("\n--- Evaluation ---")

# Evaluate Best RandomForest Model
print("\n--- Best RandomForest Model Evaluation ---")
# Predictions and probabilities are made on X_test using the best_rf_model (which includes its scaler)
predictions_rf = best_rf_model.predict(X_test)
proba_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("Accuracy (RandomForest):", accuracy_score(y_test, predictions_rf))
print("ROC AUC (RandomForest):", roc_auc_score(y_test, proba_rf))
print("Classification Report (RandomForest):\n", classification_report(y_test, predictions_rf))

# Evaluate Best GradientBoosting Model
print("\n--- Best GradientBoosting Model Evaluation ---")
predictions_gb = best_gb_model.predict(X_test)
proba_gb = best_gb_model.predict_proba(X_test)[:, 1]

print("Accuracy (GradientBoosting):", accuracy_score(y_test, predictions_gb))
print("ROC AUC (GradientBoosting):", roc_auc_score(y_test, proba_gb))
print("Classification Report (GradientBoosting):\n", classification_report(y_test, predictions_gb))

# Evaluate Ensemble Model (VotingClassifier)
print("\n--- Ensemble Model (VotingClassifier) Evaluation ---")
# Predictions and probabilities for the ensemble pipeline (which also includes its scaler)
predictions_ensemble = ensemble_pipeline.predict(X_test)
proba_ensemble = ensemble_pipeline.predict_proba(X_test)[:, 1]

print("Accuracy (Ensemble):", accuracy_score(y_test, predictions_ensemble))
print("ROC AUC (Ensemble):", roc_auc_score(y_test, proba_ensemble))
print("Classification Report (Ensemble):\n", classification_report(y_test, predictions_ensemble))


# --- Cross-Validation on the best models (for robust performance estimate) ---
print("\n--- Cross-Validation Results (on X_train, y_train) ---")

# Perform cross-validation on the full pipelines including their scalers
cv_scores_rf = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"RandomForest 5-fold Cross-Validation AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

cv_scores_gb = cross_val_score(best_gb_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"GradientBoosting 5-fold Cross-Validation AUC: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")

cv_scores_ensemble = cross_val_score(ensemble_pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Ensemble 5-fold Cross-Validation AUC: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std():.4f})")


# --- Feature Importance (from the best RandomForest model) ---
print("\n--- Feature Importance (from Best RandomForest Model) ---")
# Get feature importances from the classifier within the pipeline
feature_importances_rf = best_rf_model.named_steps['classifier'].feature_importances_
features = X.columns
importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': feature_importances_rf})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)
print(importance_df_rf)

# Plotting feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_rf.head(15))
plt.title('Top 15 Feature Importances (RandomForest)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# --- ROC Curve Plot ---
print("\n--- ROC Curve Plot ---")
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, proba_gb)
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, proba_ensemble)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {roc_auc_score(y_test, proba_rf):.2f})', color='blue')
plt.plot(fpr_gb, tpr_gb, label=f'GradientBoosting (AUC = {roc_auc_score(y_test, proba_gb):.2f})', color='green')
plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble (AUC = {roc_auc_score(y_test, proba_ensemble):.2f})', color='red', linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill (AUC = 0.50)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# --- Precision-Recall Curve Plot ---
print("\n--- Precision-Recall Curve Plot ---")
precision_rf, recall_rf, _ = precision_recall_curve(y_test, proba_rf)
precision_gb, recall_gb, _ = precision_recall_curve(y_test, proba_gb)
precision_ensemble, recall_ensemble, _ = precision_recall_curve(y_test, proba_ensemble)

plt.figure(figsize=(10, 8))
plt.plot(recall_rf, precision_rf, label=f'RandomForest (AP = {auc(recall_rf, precision_rf):.2f})', color='blue')
plt.plot(recall_gb, precision_gb, label=f'GradientBoosting (AP = {auc(recall_gb, precision_gb):.2f})', color='green')
plt.plot(recall_ensemble, precision_ensemble, label=f'Ensemble (AP = {auc(recall_ensemble, precision_ensemble):.2f})', color='red', linewidth=2, linestyle='--')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()


# --- Save the best models ---
print("\n--- Saving Best Models ---")
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
joblib.dump(best_gb_model, 'best_gradient_boosting_model.pkl')
joblib.dump(ensemble_pipeline, 'best_ensemble_model.pkl')
print("Models saved successfully: 'best_random_forest_model.pkl', 'best_gradient_boosting_model.pkl', 'best_ensemble_model.pkl'")

print("\nAdvanced diabetes prediction model training and evaluation complete!")