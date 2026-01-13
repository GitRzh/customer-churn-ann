# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import streamlit as st
# streamlit -> used to create interactive web applications

import numpy as np
# numpy -> used for numerical operations (arrays, math)

import tensorflow as tf
# tensorflow -> used to load and run the trained ANN model

import pandas as pd
# pandas -> used to create and manipulate DataFrames

import pickle
# pickle -> used to load saved encoders and scaler


# =====================
# LOAD TRAINED MODEL
# =====================

model = tf.keras.models.load_model("model.h5")
# Load the previously trained ANN model


# ============================
# LOAD ENCODERS AND SCALER
# ============================

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)
# Load OneHotEncoder for Geography feature

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)
# Load LabelEncoder for Gender feature

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
# Load StandardScaler used during training


# =========================
# STREAMLIT APP TITLE
# =========================

st.title("Customer Churn Prediction")
# Display app title on UI


# =========================
# USER INPUT FIELDS
# =========================

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)
# Dropdown for Geography values used during training

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)
# Dropdown for Gender values (Male/Female)

age = st.slider("Age", 18, 92)
# Slider for customer age

balance = st.number_input("Balance", value=0.0)
# Numeric input for account balance

credit_score = st.number_input("Credit Score", value=600)
# Numeric input for credit score

estimated_salary = st.number_input(
    "Estimated Salary",
    value=50000.0
)
# Numeric input for estimated salary

tenure = st.slider("Tenure", 0, 10)
# Slider for number of years customer stayed

num_of_products = st.slider(
    "Number of Products",
    1, 4
)
# Slider for number of bank products used

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)
# 0 -> No, 1 -> Yes

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)
# 0 -> No, 1 -> Yes


# =====================
# ENCODE CATEGORICAL DATA
# =====================

gender_encoded = label_encoder_gender.transform([gender])[0]
# Convert Gender text into numeric value


# =====================
# CREATE INPUT DATAFRAME
# =====================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})
# Create DataFrame for numerical input features


# =====================
# ONE-HOT ENCODE GEOGRAPHY
# =====================

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
# Convert Geography into one-hot encoded vector

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)
# Convert encoded geography into DataFrame


# =====================
# COMBINE ALL INPUT FEATURES
# =====================

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)
# Merge numerical and encoded categorical features


# =====================
# SCALE INPUT FEATURES
# =====================

input_data_scaled = scaler.transform(input_data)
# Apply same scaling used during model training


# =====================
# MAKE PREDICTION
# =====================

prediction = model.predict(input_data_scaled)
# Predict churn probability

prediction_proba = prediction[0][0]
# Extract probability value


# =====================
# DISPLAY RESULT
# =====================

st.write(f"### Churn Probability: {prediction_proba:.2f}")
# Display churn probability on UI

if prediction_proba > 0.5:
    st.error("The customer is likely to churn")
    # High churn risk
else:
    st.success("The customer is not likely to churn")
    # Low churn risk
