import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("🚢 Titanic Survival Prediction App")
st.write(
    "Enter passenger details below to predict survival probability based on the machine learning model."
)


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


model = load_model()

# User Input Form
st.header("Passenger Features")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=80, value=25)
    title = st.selectbox(
        "Title", ["Mr", "Miss", "Mrs", "Master", "Rare/Official"]
    )
    has_cabin = st.selectbox("Has Cabin Assigned?", ["No", "Yes"])

with col2:
    sibsp = st.number_input(
        "Siblings / Spouses Aboard", min_value=0, max_value=10, value=0
    )
    parch = st.number_input(
        "Parents / Children Aboard", min_value=0, max_value=10, value=0
    )
    fare = st.number_input(
        "Fare Paid (£)", min_value=0.0, max_value=600.0, value=32.0
    )
    embarked = st.selectbox(
        "Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
    )

# Preprocessing Inputs to match Model Features
sex_mapped = 1 if sex == "Female" else 0

title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare/Official": 5}
title_mapped = title_map[title]

embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
embarked_mapped = embarked_map[embarked]

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
is_child = 1 if age < 12 else 0
fare_per_person = fare / family_size
has_cabin_mapped = 1 if has_cabin == "Yes" else 0

# DataFrame formatted for prediction
input_data = pd.DataFrame(
    [
        {
            "Pclass": pclass,
            "Sex": sex_mapped,
            "Age": age,
            "FarePerPerson": fare_per_person,
            "Title": title_mapped,
            "FamilySize": family_size,
            "IsAlone": is_alone,
            "IsChild": is_child,
            "Embarked": embarked_mapped,
            "HasCabin": has_cabin_mapped,
        }
    ]
)

st.divider()

# Prediction button
if st.button("Predict Survival Status", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(
            f"🎉 **Result:** Passenger would have **SURVIVED**! (Confidence: {probabilities[1]*100:.1f}%)"
        )
    else:
        st.error(
            f"💀 **Result:** Passenger would **NOT HAVE SURVIVED**. (Confidence: {probabilities[0]*100:.1f}%)"
        )