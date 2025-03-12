import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and encoders
model = load_model('model.h5')

# Load the encoders and scaler
with open('lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('geo_encoder.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", lable_encoder_gender.classes_)  # Fixed `.classes_`
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)  # Fixed column name
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data (ensure correct feature names)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],  # Placeholder for one-hot encoding
    'Gender': [lable_encoder_gender.transform([gender])[0]],  # Encode gender
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],  # Ensure correct feature name
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode geography (one-hot encoding)
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

# Drop the original geography column and add encoded values
input_data = input_data.drop(columns=['Geography'])
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Ensure column order matches the training phase
expected_features = scaler.feature_names_in_  # Retrieves correct order
input_data = input_data[expected_features]  # Reorder columns

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display result
if prediction_proba < 0.5:
    st.write("The customer is **likely to churn**.")
else:
    st.write("The customer is **likely to stay**.")
