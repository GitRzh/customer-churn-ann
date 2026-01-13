# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import pandas as pd
# pandas -> used for loading and manipulating tabular data (CSV, DataFrame)

import pickle
# pickle -> used to save trained encoders/scalers for future use

import datetime
# datetime -> used to create timestamped TensorBoard logs

from sklearn.model_selection import train_test_split
# train_test_split -> splits dataset into training and testing sets

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# StandardScaler -> feature scaling (mean=0, std=1)
# LabelEncoder -> converts binary categorical labels into numbers
# OneHotEncoder -> converts multi-class categorical data into binary columns

import tensorflow as tf
# TensorFlow -> deep learning framework

from tensorflow.keras.models import Sequential
# Sequential -> model where layers are stacked linearly

from tensorflow.keras.layers import Dense
# Dense -> fully connected neural network layer

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# EarlyStopping -> stops training when validation loss stops improving
# TensorBoard -> visualizes training metrics


# =================
# LOAD THE DATASET
# =================

data = pd.read_csv("Churn_Modelling.csv")
# Load the customer churn dataset into a DataFrame

data.head()
# Display first few rows to understand dataset structure


# ==========================
# DROP IRRELEVANT COLUMNS
# ==========================

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# These columns do not help prediction, so remove them


# =============================
# ENCODE CATEGORICAL FEATURES
# =============================

# Encode Gender column (Male/Female -> 0/1)
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
# Converts text labels into numeric values

# One-hot encode Geography column (France/Germany/Spain -> multiple binary columns)
onehot_encoder_geo = OneHotEncoder()
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
# Convert Geography into binary vectors

# Convert encoded geography into DataFrame with column names
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine encoded geography columns with original dataset
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
# Remove original Geography column and add encoded columns


# ============================
# SAVE ENCODERS FOR DEPLOYMENT
# ============================

with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)
# Save gender encoder for consistent future predictions

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)
# Save geography encoder for deployment use


# ============================
# SPLIT FEATURES AND TARGET
# ============================

X = data.drop('Exited', axis=1)
# X -> independent variables (input features)

y = data['Exited']
# y -> dependent variable (customer churn: 0 or 1)


# ====================
# TRAIN-TEST SPLIT
# ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Split data into 80% training and 20% testing
# random_state ensures reproducibility


# =================
# FEATURE SCALING
# =================

scaler = StandardScaler()
# Create StandardScaler object

X_train = scaler.fit_transform(X_train)
# Fit scaler on training data and scale it

X_test = scaler.transform(X_test)
# Scale test data using same scaler

# Save scaler for deployment
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# ==================
# BUILD ANN MODEL
# ==================

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Hidden layer 1 -> learns complex patterns

    Dense(32, activation='relu'),
    # Hidden layer 2 -> refines learned features

    Dense(1, activation='sigmoid') #or SoftMax
    # Try to change values
    # Output layer -> sigmoid for binary classification
])

model.summary()
# Display model architecture


# ==========================
# OPTIMIZER & LOSS FUNCTION
# ==========================

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# Adam optimizer -> adaptive learning rate

loss = tf.keras.losses.BinaryCrossentropy()
# BinaryCrossentropy -> suitable for binary classification


# ==================
# COMPILE THE MODEL
# ==================

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy']
)
# Configure model for training


# =====================
# TENSORBOARD SETUP
# =====================

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Create unique log directory for TensorBoard

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)
# Enable visualization of weights and metrics


# =================
# EARLY STOPPING
# =================

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
# Stop training if validation loss doesn't improve


# ==================
# TRAIN THE MODEL
# ==================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping_callback]
)
# Train ANN model with validation monitoring


# ======================
# SAVE TRAINED MODEL
# ======================

model.save('model.h5')
# Save trained ANN model for future use