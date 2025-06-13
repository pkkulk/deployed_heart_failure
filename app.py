import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and training columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # This should be the final df.columns after get_dummies()

st.title("❤️ Heart Disease Risk Prediction App")

# User inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert inputs to dataframe
input_data = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": 1 if fasting_bs == "Yes" else 0,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}

input_df = pd.DataFrame([input_data])

# One-hot encode + reindex (matches training)
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# Predict
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

# Output
st.subheader("Prediction Result:")
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f} probability)")
else:
    st.success(f"✅ Low Risk of Heart Disease ({1 - prob:.2f} probability)")
