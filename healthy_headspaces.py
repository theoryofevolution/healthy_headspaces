import streamlit as st
import numpy as np
import joblib

# Load the model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Display the logo at the top of the page
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn.discordapp.com/attachments/844331468544016414/1335160162331394119/Copy_of_HEALTHY_HEADSPACE_Logo_1.png?ex=679f284c&is=679dd6cc&hm=2611beb65f2607f9e0f22f16acbd1b12a2fc5e2ff78bc33d95de531c2241f039&" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("Healthy Headspaces - Depression Prediction Applet")

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
    if prediction[0] == 1:
        st.subheader("🟣 Depression Detected")
        st.markdown(
            """
            ### 💡 Resources for Mental Health Support
            - **📞 Helplines (USA)**:
              - National Suicide Prevention Lifeline: [988](https://988lifeline.org)
              - Crisis Text Line: Text **HELLO** to 741741
              - The Trevor Project (LGBTQ+): [TheTrevorProject.org](https://www.thetrevorproject.org/)

            - **🌍 International Resources**:
              - Find your country’s helpline: [Check Here](https://www.findahelpline.com/)
              
            - **📖 Self-Help & Wellness**:
              - Mental health exercises: [Mind.org](https://www.mind.org.uk/)
              - Free guided meditation: [Headspace](https://www.headspace.com/)
              - Student mental health tips: [Active Minds](https://www.activeminds.org/)
            
            If you're struggling, please reach out. You're not alone. 💜
            """
        )
    else:
        st.subheader("✅ No Depression Detected")

# Footer (Fixed at the bottom)
st.markdown(
    """
    <style>
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
<div class='footer'>Made with ❤ by People of Programming |© 2025 People of Programming | <a href="https://peopleofprogramming.com" target="_blank" style="color: #54207E; text-decoration: none;">Visit Our Website</a></div>
    """,
    unsafe_allow_html=True
)
