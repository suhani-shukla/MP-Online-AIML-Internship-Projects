"""
app.py
Streamlit web app for the Adult Income Classification project.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from predict import predict_income

st.set_page_config(page_title="Adult Income Classifier", page_icon="💰", layout="centered")

st.title("💰 Adult Income Classification")
st.write(
    "Predict whether a person's annual income exceeds **$50K** "
    "based on census attributes, using a trained Random Forest model."
)

st.divider()

# --- Input form ---
with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 30)
        education_num = st.slider("Education (years)", 1, 16, 10)
        hours_per_week = st.slider("Hours per week", 1, 99, 40)
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
        sex = st.selectbox("Sex", ["Male", "Female"])

    with col2:
        workclass = st.selectbox("Workclass", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated",
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv",
            "Armed-Forces"
        ])
        relationship = st.selectbox("Relationship", [
            "Wife", "Own-child", "Husband", "Not-in-family",
            "Other-relative", "Unmarried"
        ])
        race = st.selectbox("Race", [
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
        ])
        native_country = st.selectbox("Native Country", [
            "United-States", "Mexico", "Philippines", "Germany", "India",
            "China", "England", "Canada", "Other"
        ])

    submitted = st.form_submit_button("Predict Income Class")

# --- Prediction ---
if submitted:
    input_data = {
        "age": age,
        "workclass": workclass,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    try:
        result = predict_income(input_data)

        st.divider()
        label = result["predicted_income"]
        proba_gt50k = result["probability_gt_50k"]

        if label == ">50K":
            st.success(f"### Predicted Income: **{label}**")
        else:
            st.info(f"### Predicted Income: **{label}**")

        st.metric("Probability of earning >$50K", f"{proba_gt50k*100:.1f}%")
        st.progress(proba_gt50k)

        with st.expander("See input summary"):
            st.dataframe(pd.DataFrame([input_data]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Model: Random Forest • Dataset: UCI Adult / Census Income")