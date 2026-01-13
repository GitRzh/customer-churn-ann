# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import tensorflow as tf
# tensorflow -> used to load and run the trained ANN model

from tensorflow.keras.models import load_model
# load_model -> loads saved Keras model (.h5)

import pickle
# pickle -> used to load saved encoders and scaler

import pandas as pd
# pandas -> used to create and manipulate DataFrames

import numpy as np
# numpy -> used for numerical operations


# =====================
# LOAD TRAINED MODEL
# =====================

model = load_model("model.h5")
# Load the previously trained churn prediction model


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


# =====================
# RAW INPUT DATA
# =====================

input_data = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 700000
}
# Dictionary containing raw customer details


# =====================
# ONE-HOT ENCODE GEOGRAPHY
# =====================

geo_encoded = onehot_encoder_geo.transform(
    [[input_data["Geography"]]]
).toarray()
# Convert Geography into one-hot encoded vector

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)
# Convert encoded geography into DataFrame


# =====================
# ENCODE GENDER
# =====================

input_data["Gender"] = label_encoder_gender.transform(
    [input_data["Gender"]]
)[0]
# Convert Gender from text to numeric value


# =====================
# CREATE INPUT DATAFRAME
# =====================

input_df = pd.DataFrame([input_data])
# Convert input dictionary into DataFrame


# =====================
# MERGE ENCODED FEATURES
# =====================

input_df = pd.concat(
    [input_df.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)
# Remove original Geography column and add encoded columns


# =====================
# SCALE INPUT FEATURES
# =====================

input_scaled = scaler.transform(input_df)
# Apply same feature scaling used during training


# =====================
# MAKE PREDICTION
# =====================

prediction = model.predict(input_scaled)
# Predict churn probability

prediction_proba = prediction[0][0]
# Extract probability value


# =====================
# DISPLAY OUTPUT
# =====================

if prediction_proba > 0.5:
    print("The customer is likely to churn")
    # High churn probability
else:
    print("The customer is not likely to churn")
    # Low churn probability
