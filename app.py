import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 50.0)
total = st.number_input("Total Charges", 0.0, 50000.0, 500.0)
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])

input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'gender': 1 if gender == "Male" else 0,
    'Partner': 1 if partner == "Yes" else 0
}

for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

df = pd.DataFrame([input_data], columns=feature_names)
df_scaled = scaler.transform(df)

if st.button("Predict"):
    result = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)[0][1]
    st.write("Churn:", "Yes" if result[0] == 1 else "No")
    st.write("Probability:", round(prob * 100, 2), "%")
