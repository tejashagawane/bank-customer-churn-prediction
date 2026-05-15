import streamlit as st
import joblib
import pandas as pd
from utils.helper import preprocess_input

st.set_page_config(page_title="Churn AI System", layout="wide")

st.title("🏦 AI-Powered Customer Churn Prediction System")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================
# INPUT FORM
# =========================
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit = st.slider("Credit Score", 300, 850, 600)
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

with col2:
    products = st.slider("Products", 1, 4, 1)
    card = st.selectbox("Credit Card", [1, 0])
    active = st.selectbox("Active Member", [1, 0])
    salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)

geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict"):

    user_data = {
        "CreditScore": credit,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "Geography": geo,
        "Gender": gender
    }

    # Preprocess input
    input_data = preprocess_input(user_data)

    # Ensure exact feature order used during training
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    # Scale
    scaled = scaler.transform(input_data)

    # Predict
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("📊 Result")

    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.progress(float(prob))
    st.write(f"**Churn Probability:** {prob:.2f}")

    # =========================
    # REPORT DOWNLOAD
    # =========================
    report = f"""
    Customer Churn Report

    Credit Score: {credit}
    Age: {age}
    Geography: {geo}
    Probability: {prob:.2f}
    """

    st.download_button(
        "📄 Download Report",
        report,
        "report.txt"
    )