import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('flood_model.pkl')

# Title for the web app
st.title("Flood Impact Prediction")

# Input fields for user to enter feature values
st.header("Enter flood event details")

duration = st.number_input("Duration (Days)", min_value=0, step=1)
fatalities = st.number_input("Human Fatality Count", min_value=0, step=1)
severity = st.number_input("Severity (0 to 10 scale)", min_value=0.0, max_value=10.0, step=0.1)

# Prediction button
if st.button("Predict Impact"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[duration, fatalities, severity]], columns=['Duration(Days)', 'Human fatality', 'Severity'])
    
    # Predict the impact
    prediction = model.predict(input_data)[0]
    
    # Display the result
    if prediction == 1:
        st.success("Significant impact predicted.")
    else:
        st.info("No significant impact predicted.")
