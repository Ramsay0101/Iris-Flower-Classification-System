import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title("Iris Flower Classification")

# Input fields for feature values
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Button to make prediction
if st.button("Classify"):
    # Prepare input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Apply the same scaling as during model training
    input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.success(f"The predicted species is: {prediction[0]}")

