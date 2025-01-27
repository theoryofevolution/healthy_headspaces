import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the pre-trained model and encoders
# Save your model and scaler using `joblib.dump(model, "model.pkl")` and `joblib.dump(scaler, "scaler.pkl")` beforehand
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# App Title
st.title("Depression Prediction App")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, step=1)
academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
study_hours = st.number_input("Study Hours", min_value=0, max_value=24, step=1)
financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Encoding user inputs
inputs = [
    label_encoders["Gender"].transform([gender])[0],
    age,
    academic_pressure,
    study_satisfaction,
    label_encoders["Sleep Duration"].transform([sleep_duration])[0],
    label_encoders["Dietary Habits"].transform([dietary_habits])[0],
    label_encoders["Have you ever had suicidal thoughts ?"].transform([suicidal_thoughts])[0],
    study_hours,
    financial_stress,
    label_encoders["Family History of Mental Illness"].transform([family_history])[0]
]

# Scaling the inputs
scaled_inputs = scaler.transform([inputs])

# Predicting
if st.button("Predict"):
    prediction = model.predict(scaled_inputs)
    result = "Depression Detected" if prediction[0] == 1 else "No Depression Detected"
    st.subheader(result)
