"""
streamlit_app.py
------------------
Main application. Implements the full user journey discussed earlier:

Landing page -> Select disease -> Fill health parameters -> Predict ->
See risk result -> Chat with AI for follow-up advice.

Run with: streamlit run streamlit_app.py

IMPORTANT: This expects trained models to already exist in ../models/
(see models/README.md for how to generate them).
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Use a path based on this script's own location (not the current working
# directory) so imports work correctly no matter where streamlit is launched from.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from preprocessing import load_scaler
import chatbot

st.set_page_config(page_title="AI Health Risk Assistant", page_icon="🩺", layout="centered")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ---------------------------------------------------------------------
# Model loading (cached so it only happens once per session, not per click)
# ---------------------------------------------------------------------

@st.cache_resource
def load_heart_assets():
    import joblib
    model = load_model(os.path.join(MODELS_DIR, "heart_model.h5"))
    scaler = load_scaler(os.path.join(MODELS_DIR, "heart_scaler.pkl"))
    columns = joblib.load(os.path.join(MODELS_DIR, "heart_columns.pkl"))
    return model, scaler, columns


@st.cache_resource
def load_diabetes_assets():
    model = load_model(os.path.join(MODELS_DIR, "diabetes_model.h5"))
    scaler = load_scaler(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))
    return model, scaler


@st.cache_resource
def load_kidney_assets():
    import joblib
    model = load_model(os.path.join(MODELS_DIR, "kidney_model.h5"))
    scaler = load_scaler(os.path.join(MODELS_DIR, "kidney_scaler.pkl"))
    columns = joblib.load(os.path.join(MODELS_DIR, "kidney_columns.pkl"))
    return model, scaler, columns


# ---------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------

st.title("🩺 AI Health Risk Assistant")
st.markdown(
    "Predict your risk for **Heart Disease**, **Diabetes**, or **Chronic Kidney Disease** "
    "using machine learning models trained on clinical datasets."
)
st.info(
    "⚠️ **Disclaimer**: This tool is for educational purposes only and is NOT a substitute "
    "for professional medical advice, diagnosis, or treatment. Always consult a qualified "
    "doctor for any health concerns.",
    icon="⚠️",
)

st.divider()

# ---------------------------------------------------------------------
# Step 1: Disease selection
# ---------------------------------------------------------------------

disease = st.selectbox(
    "Select what you want to check:",
    ["Heart Disease", "Diabetes", "Chronic Kidney Disease"],
)

st.divider()

# Initialize session state for prediction results - this is what keeps the
# result and chat history visible even after Streamlit reruns the script
# (which happens on every chat message sent).
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
    st.session_state.prediction_probability = None
    st.session_state.prediction_disease = None
    st.session_state.chat_history = []

result = st.session_state.prediction_result
probability = st.session_state.prediction_probability

if disease == "Heart Disease":
    st.subheader("Enter your health parameters")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [True, False])
        restecg = st.selectbox(
            "Resting ECG Result", ["normal", "lv hypertrophy", "st-t abnormality"]
        )
    with col2:
        thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina?", [True, False])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0.0, 1.0, 2.0, 3.0])
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    if st.button("Predict Heart Disease Risk", type="primary"):
        model, scaler, expected_columns = load_heart_assets()

        # Build raw input matching the original dataset's raw columns/values,
        # then one-hot encode and align to the exact columns seen during training
        # (same approach as the kidney disease section below).
        raw_input = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }
        raw_df = pd.DataFrame([raw_input])
        encoded_df = pd.get_dummies(raw_df)

        # Align columns with what the model was trained on
        encoded_df = encoded_df.reindex(columns=expected_columns, fill_value=0)

        input_scaled = scaler.transform(encoded_df)
        probability = float(model.predict(input_scaled, verbose=0)[0][0])
        result = "High Risk" if probability > 0.5 else "Low Risk"

        st.session_state.prediction_result = result
        st.session_state.prediction_probability = probability
        st.session_state.prediction_disease = disease
        st.session_state.chat_history = []  # reset chat for a fresh prediction

elif disease == "Diabetes":
    st.subheader("Enter your health parameters")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 110)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin Level", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age_d = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes Risk", type="primary"):
        input_dict = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age_d,
        }
        input_df = pd.DataFrame([input_dict])

        model, scaler = load_diabetes_assets()
        input_scaled = scaler.transform(input_df)
        probability = float(model.predict(input_scaled, verbose=0)[0][0])
        result = "High Risk" if probability > 0.5 else "Low Risk"

        st.session_state.prediction_result = result
        st.session_state.prediction_probability = probability
        st.session_state.prediction_disease = disease
        st.session_state.chat_history = []

else:  # Chronic Kidney Disease
    st.subheader("Enter your health parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_k = st.number_input("Age", 1, 120, 45)
        bp = st.number_input("Blood Pressure (mm Hg)", 40, 200, 80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
        su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    with col2:
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
        bgr = st.number_input("Blood Glucose Random", 20, 500, 120)
        bu = st.number_input("Blood Urea", 1, 400, 40)
        sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.0, step=0.1)
        sod = st.number_input("Sodium", 100, 200, 135)
        pot = st.number_input("Potassium", 1.0, 15.0, 4.5, step=0.1)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
    with col3:
        wc = st.number_input("White Blood Cell Count", 2000, 25000, 8000)
        rc = st.number_input("Red Blood Cell Count", 2.0, 8.0, 5.0, step=0.1)
        htn = st.selectbox("Hypertension?", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus?", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease?", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema?", ["no", "yes"])
        ane = st.selectbox("Anemia?", ["no", "yes"])

    if st.button("Predict Kidney Disease Risk", type="primary"):
        model, scaler, expected_columns = load_kidney_assets()

        # Build raw input matching original dataset's raw columns/values,
        # then one-hot encode and align to the exact columns seen during training.
        raw_input = {
            "age": age_k, "bp": bp, "sg": sg, "al": al, "su": su,
            "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
            "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
            "htn": htn, "dm": dm, "cad": cad,
            "appet": appet, "pe": pe, "ane": ane,
        }
        raw_df = pd.DataFrame([raw_input])
        encoded_df = pd.get_dummies(raw_df)

        # Align columns with what the model was trained on - fill missing
        # dummy columns with 0, drop any extras, keep exact training order
        encoded_df = encoded_df.reindex(columns=expected_columns, fill_value=0)

        input_scaled = scaler.transform(encoded_df)
        probability = float(model.predict(input_scaled, verbose=0)[0][0])
        result = "High Risk" if probability > 0.5 else "Low Risk"

        st.session_state.prediction_result = result
        st.session_state.prediction_probability = probability
        st.session_state.prediction_disease = disease
        st.session_state.chat_history = []


# ---------------------------------------------------------------------
# Step 3: Show result
# ---------------------------------------------------------------------

result = st.session_state.prediction_result
probability = st.session_state.prediction_probability
disease = st.session_state.prediction_disease or disease

if result is not None:
    st.divider()
    st.subheader("Result")

    if result == "High Risk":
        st.error(f"**{result}** — predicted probability: {probability:.0%}")
    else:
        st.success(f"**{result}** — predicted probability: {probability:.0%}")

    st.progress(probability)

    # ------------------------------------------------------------
    # Step 4: Chatbot follow-up
    # ------------------------------------------------------------
    st.divider()
    st.subheader("💬 Ask the AI Health Assistant")
    st.caption("Ask about lifestyle changes, what the result means, or general next steps.")

    if not st.session_state.chat_history:
        st.session_state.chat_history = []
        if not os.environ.get("GEMINI_API_KEY"):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⚠️ Chatbot is currently unavailable because GEMINI_API_KEY is not set. "
                            "The prediction above still works normally - set the environment variable "
                            "and restart the app to enable follow-up chat."
            })
        else:
            try:
                opening = chatbot.get_initial_message(disease, result, probability)
                st.session_state.chat_history.append({"role": "assistant", "content": opening})
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Hi! I couldn't connect to the chat service right now. ({e})"
                })

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Type your question here...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                reply = chatbot.get_followup_response(disease, result, probability, user_msg)
            except Exception as e:
                reply = f"Sorry, I couldn't reach the chat service right now. ({e})"
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
