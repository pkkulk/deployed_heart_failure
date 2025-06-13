import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")  # list of columns used during training

st.title("💓 Heart Disease Prediction App")

# User input form
def user_input():
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    trestbps = st.slider("Resting BP", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("FastingBS > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment", ["Up", "Flat", "Down"])

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": trestbps,
        "Cholesterol": chol,
        "FastingBS": 1 if fbs == "Yes" else 0,
        "RestingECG": restecg,
        "MaxHR": thalach,
        "ExerciseAngina": 1 if exang == "Yes" else 0,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }
    return pd.DataFrame([data])

# Process input
input_df = user_input()

# Match training encoding
full_df = pd.get_dummies(input_df)
full_df = full_df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(full_df)[0]
    proba = model.predict_proba(full_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Low risk of Heart Disease. (Confidence: {1 - proba:.2f})")
