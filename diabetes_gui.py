import pandas as pd
import numpy as np
import gradio as gr
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load model and columns
best_ensemble_model = joblib.load('best_ensemble_model.pkl')
model_feature_columns = joblib.load('model_feature_columns.pkl')

# Define BMI categories
all_bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree_function, age):
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    input_df = pd.DataFrame([input_data])

    input_df['BMI_Category'] = pd.cut(input_df['BMI'],
                                      bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                                      labels=all_bmi_labels,
                                      right=False)

    temp_bmi_dummies = pd.DataFrame(0, index=input_df.index, columns=[f'BMI_Cat_{label}' for label in all_bmi_labels])
    if not input_df['BMI_Category'].isna().iloc[0]:
        category_col_name = f'BMI_Cat_{input_df["BMI_Category"].iloc[0]}'
        if category_col_name in temp_bmi_dummies.columns:
            temp_bmi_dummies[category_col_name] = 1

    input_df = pd.concat([input_df.drop('BMI_Category', axis=1), temp_bmi_dummies], axis=1)

    input_df['Glucose_Insulin_Ratio'] = input_df['Glucose'] / (input_df['Insulin'] + 1e-6)
    input_df['BloodPressure_Age_Interaction'] = input_df['BloodPressure'] * input_df['Age']
    input_df['Has_Insulin'] = (input_df['Insulin'] > 0).astype(int)

    final_input_df = pd.DataFrame(columns=model_feature_columns)
    for col in model_feature_columns:
        final_input_df[col] = input_df[col] if col in input_df.columns else 0
    final_input_df = final_input_df.iloc[0].to_frame().T

    prediction_proba = best_ensemble_model.predict_proba(final_input_df)[:, 1]
    prediction = (prediction_proba > 0.5).astype(int)

    if prediction[0] == 1:
        return f"**Diabetic** (Probability: {prediction_proba[0]:.2f})"
    else:
        return f"**Non-Diabetic** (Probability: {prediction_proba[0]:.2f})"

# Use Blocks for better layout and animation support
with gr.Blocks(title="Diabetes Prediction", theme=gr.themes.Base(), css="") as demo:
    gr.Markdown("# Diabetes Prediction Model")
    gr.Markdown("Enter patient details below. The model will predict diabetes risk")

    with gr.Row():
        pregnancies = gr.Slider(0, 17, step=1, label="Pregnancies", value=1, interactive=True)
        glucose = gr.Number(label="Glucose", value=120)
        blood_pressure = gr.Number(label="Blood Pressure", value=70)
        skin_thickness = gr.Number(label="Skin Thickness", value=20)

    with gr.Row():
        insulin = gr.Number(label="Insulin", value=80)
        bmi = gr.Number(label="BMI", value=30)
        dpf = gr.Number(label="Diabetes Pedigree Function", value=0.5)
        age = gr.Number(label="Age", value=30)

    submit_btn = gr.Button("🔍 Predict", variant="primary", scale=2)
    output = gr.Markdown("")

    examples = gr.Examples(
        examples=[
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31]
        ],
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age]
    )

    submit_btn.click(fn=predict_diabetes,
                     inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age],
                     outputs=output)

demo.launch(share=True)
