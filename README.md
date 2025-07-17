Diabetes Prediction using Machine Learning
Project Overview
This project aims to develop a robust machine learning model to predict the likelihood of an individual having diabetes based on various health metrics. Leveraging the Pima Indians Diabetes Database, the project encompasses a complete machine learning workflow from data preprocessing and feature engineering to model training, hyperparameter optimization, ensemble modeling, and comprehensive evaluation. The ultimate goal is to assist healthcare professionals in early identification of at-risk individuals, enabling timely intervention and improved patient outcomes.

Features
Data Preprocessing:

Handling of zero values in critical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) by treating them as missing values and imputing with the mean.

Outlier handling using the Interquartile Range (IQR) method (Winsorization/clipping).

Feature Engineering:

Creation of BMI_Category (Underweight, Normal, Overweight, Obese I, II, III).

Calculation of Glucose_Insulin_Ratio to infer insulin resistance.

Generation of interaction terms like BloodPressure_Age_Interaction, Pregnancy_Glucose_Interaction, and Age_BMI_Interaction.

Binary feature Has_Insulin indicating insulin presence.

One-hot encoding for categorical features.

Model Training & Optimization:

Implementation of two powerful classification algorithms: Random Forest Classifier and Gradient Boosting Classifier.

Utilizing sklearn.pipeline for streamlined data transformation and model training.

Extensive Hyperparameter Optimization using RandomizedSearchCV with roc_auc as the scoring metric for both models.

Ensemble Modeling:

A VotingClassifier combines the strengths of the optimized Random Forest and Gradient Boosting models using 'soft' voting for improved predictive performance and robustness.

Comprehensive Evaluation:

Reporting of key metrics: Accuracy, ROC AUC, Precision, Recall, and F1-Score through detailed classification reports.

Cross-validation (5-fold) for robust performance estimation.

Visualization of Feature Importances from the best Random Forest model.

Plotting of Receiver Operating Characteristic (ROC) curves and Precision-Recall curves for visual comparison of model performance.

Model Persistence:

Saving the best-performing individual and ensemble models using joblib for future deployment and inference.

Saving the exact feature columns used during training (model_feature_columns.pkl) to ensure consistency during deployment.

Dataset
This project utilizes the Pima Indians Diabetes Database, a publicly available dataset from the UCI Machine Learning Repository.

Source: UCI Machine Learning Repository

Number of Observations: 768

Number of Features: 8

Target Variable: Outcome (0 for non-diabetic, 1 for diabetic)

Features include: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

Installation and Setup
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

(Replace YOUR_USERNAME and YOUR_REPOSITORY_NAME with your actual GitHub username and repository name)

Create a virtual environment (recommended):

python -m venv venv
# On Windows
.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

(If requirements.txt is not present, you can generate it after running the script once, or manually install the listed libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib)

Download the dataset:

Ensure you have the diabetes.csv file in the root directory of your project. You can typically find this dataset on Kaggle or the UCI Machine Learning Repository.

Usage
To train the models, perform evaluation, and save the best models, simply run the main Python script:

python diabetes_prediction.py

The script will:

Perform data preprocessing and feature engineering.

Train and optimize Random Forest and Gradient Boosting models.

Train an Ensemble (VotingClassifier) model.

Print evaluation metrics (Accuracy, ROC AUC, Classification Report) for all models to the console.

Display plots for Feature Importance, ROC Curves, and Precision-Recall Curves.

Save the trained models (.pkl files) and the feature columns list.

Project Structure
.
├── diabetes.csv              # The dataset file
├── diabetes_prediction.py    # Main script for data processing, training, and evaluation
├── best_random_forest_model.pkl # Saved best Random Forest model
├── best_gradient_boosting_model.pkl # Saved best Gradient Boosting model
├── best_ensemble_model.pkl   # Saved best Ensemble model
├── model_feature_columns.pkl # List of feature columns used for training
├── requirements.txt          # Python dependencies
└── README.md                 # This file

Results and Evaluation Summary
Upon running the script, you will see detailed evaluation metrics printed to the console. Here's an example of the typical output for accuracy and ROC AUC:

--- Best RandomForest Model Evaluation ---
Accuracy (RandomForest): 0.7727
ROC AUC (RandomForest): 0.8258

--- Best GradientBoosting Model Evaluation ---
Accuracy (GradientBoosting): 0.7727
ROC AUC (GradientBoosting): 0.8351

--- Ensemble Model (VotingClassifier) Evaluation ---
Accuracy (Ensemble): 0.7662
ROC AUC (Ensemble): 0.8360

(Note: Actual values may vary slightly based on random_state and environment.)

The project demonstrates that ensemble methods, combining optimized individual models, often yield strong and robust predictive performance for diabetes detection.

Future Work
Data Expansion: Incorporate additional datasets from diverse populations and longitudinal data to enhance model robustness and applicability.

Advanced Algorithms: Explore more advanced machine learning algorithms, such as deep learning models (e.g., ANNs, LSTMs), for potential further improvements.

Feature Engineering: Continue to experiment with new feature engineering techniques to uncover more complex relationships within the data.

Real-World Testing: Conduct field testing of the deployed model in clinical settings to gather feedback and assess its practical utility.

User Feedback: Collect feedback from healthcare providers and patients to refine the model and user interface for better usability.

Ethical and Privacy Considerations: Continuously monitor the model for biases and ensure strict adherence to data privacy regulations (e.g., HIPAA, GDPR).

Model Interpretability: Explore techniques like SHAP or LIME to better understand model predictions and increase trust.

Contributors

Mahmoud Mohamed Elsawy

Eyad Ibrahim Hejab

Ahmed Ehab Mokhtar

Keroles Ehab

License
This project is licensed under the AMIT License.
