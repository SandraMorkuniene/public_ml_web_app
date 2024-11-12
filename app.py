
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel

# Define the Patient class
class Patient(BaseModel):
    gender: str
    age: float
    hypertension: float
    heart_disease: float
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

# Load your trained XGBoost model
model = joblib.load("best_xgboost_model.joblib")

# Streamlit app title and description
st.title("Stroke Risk Predictor")

st.write("""
This app predicts the risk of stroke based on your data.
Please provide the required information below.
""")

# User input fields corresponding to Patient attributes
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
heart_disease = st.selectbox("Heart Disease (1 = Yes, 0 = No)", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["Never_smoked", "Formerly_smoked", "Smokes", "Unknown"])

# Collect user data in a dictionary
user_input = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}

# Display user data
st.write("### Patient Data")
st.json(user_input)

# Convert user input to a Patient instance
try:
    patient = Patient(**user_input)
except Exception as e:
    st.error(f"Error in patient data: {e}")

# Prediction button
if st.button("Predict Stroke Risk"):
    try:
        # Convert the Patient instance to a DataFrame for the model
        patient_df = pd.DataFrame([patient.dict()])

        # Make predictions
        prediction = model.predict(patient_df)[0]
        prediction_prob = model.predict_proba(patient_df)[0, 1]

        # Display the result
        if prediction == 1:
            st.error(f"High risk of stroke. Probability: {prediction_prob:.2f}")
        else:
            st.success(f"Low risk of stroke. Probability: {prediction_prob:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
