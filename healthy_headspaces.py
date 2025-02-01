import streamlit as st
import numpy as np
import joblib

# Load the model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #C1F0FC; /* Light Blue Background */
            color: #54207E; /* Dark Purple Text */
        }
        .stApp {
            background-color: #C1F0FC;
        }
        .stButton>button {
            background-color: #A260D7; /* Light Purple Buttons */
            color: white;
            border-radius: 10px;
            border: 2px solid #54207E;
            font-size: 18px;
        }
        .stTextInput, .stSelectbox, .stNumberInput, .stSlider {
            color: #54207E;
            background-color: white;
            border-radius: 5px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #54207E; /* Dark Purple Titles */
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #54207E;
            background-color: #C1F0FC;
        }
    </style>
    <div class='footer'>© 2025 People of Programming</div>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("💜 Healthy Headspaces - Depression Prediction Applet")

# Brief Report on Depression in Students
st.header("Understanding Depression Among Students")
st.markdown(
    """
    Depression is a significant concern among students, affecting their academic performance, social interactions, and overall well-being. Recent studies have highlighted the prevalence and impact of depression in educational settings:

    - **Prevalence**: A survey conducted across 133 college campuses during 2021–2022 found that 44% of students reported symptoms of depression, with 15% seriously considering suicide in the past year. [Source: Mayo Clinic Health System](https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/college-students-and-depression)

    - **Impact**: Mental health challenges can lead to reduced quality of life, academic difficulties, and physical health issues. Addressing these concerns is crucial for students' success and long-term well-being. [Source: Suicide Prevention Resource Center](https://sprc.org/settings/colleges-and-universities/consequences-of-student-mental-health-issues/)

    - **Trends**: While rates of anxiety and depression are at all-time highs, more college students than ever before are seeking therapy or counseling, indicating a positive shift towards addressing mental health. [Source: University of Michigan School of Public Health](https://sph.umich.edu/news/2023posts/college-students-anxiety-depression-higher-than-ever-but-so-are-efforts-to-receive-care.html)

    Understanding these statistics underscores the importance of early detection and intervention. Tools like this applet aim to assist in identifying potential signs of depression, encouraging timely support and resources.
    """
)

# User Inputs
st.header("Input Your Information")
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
    result = "🟣 Depression Detected" if prediction[0] == 1 else "✅ No Depression Detected"
    st.subheader(result)
