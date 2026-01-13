import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# Load model
model = load_model("model.h5")

# Load encoders & scaler
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Input data
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

# Encode Geography
geo_encoded = onehot_encoder_geo.transform(
    [[input_data["Geography"]]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Encode Gender
input_data["Gender"] = label_encoder_gender.transform(
    [input_data["Gender"]]
)[0]

# Create input dataframe
input_df = pd.DataFrame([input_data])

# Merge one-hot encoded geography
input_df = pd.concat(
    [input_df.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba > 0.5:
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")