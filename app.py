import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset (replace with your actual dataset file or remove for production)
data = pd.read_csv('health_data.csv')

# Separate features and target variable
X = data[['pulse', 'body temperature', 'SpO2']]
y = data['Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app interface
st.title("Health Status Prediction API")

# Inputs for API
st.subheader("Send Your Data:")
pulse = st.number_input("Pulse Rate", min_value=0.0, step=0.1, format="%.1f")
body_temp = st.number_input("Body Temperature (°C)", min_value=0.0, step=0.1, format="%.1f")
spo2 = st.number_input("SpO2 Level (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")

# Button to predict
if st.button("Predict"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[pulse, body_temp, spo2]], columns=['pulse', 'body temperature', 'SpO2'])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Map prediction to human-readable labels (adjust based on your dataset)
    label_mapping = {
        0: "Healthy",
        1: "Non-life-threatening",
        2: "Life-threatening"
    }
    prediction_label = label_mapping.get(prediction[0], "Unknown")
    
    # Display the prediction
    st.write(f"Predicted Health Status: **{prediction_label}**")