import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and features
model = joblib.load('construction_cost_model.pkl')
model_features = joblib.load('model_features.pkl')

# Streamlit app title
st.title("Construction Project Cost Predictor")

# User inputs
project_type = st.selectbox("Project Type", ["Residential", "Commercial"])
num_workers = st.number_input("Number of Workers", min_value=5, max_value=100, value=20)
materials = st.selectbox("Main Materials", ["Wood", "Steel", "Concrete"])
project_length = st.number_input("Project Length (Months)", min_value=2, max_value=24, value=6)

# Process user inputs
def preprocess_input(project_type, num_workers, materials, project_length):
    input_data = pd.DataFrame({
        'Num_Workers': [num_workers],
        'Project_Length': [project_length],
        'Project_Type_Residential': [1 if project_type == 'Residential' else 0],
        'Project_Type_Commercial': [1 if project_type == 'Commercial' else 0],
        'Materials_Wood': [1 if materials == 'Wood' else 0],
        'Materials_Steel': [1 if materials == 'Steel' else 0],
        'Materials_Concrete': [1 if materials == 'Concrete' else 0],
    })

    # Ensure all model features are present
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the model
    input_data = input_data[model_features]

    return input_data

# Predict cost
if st.button("Predict Cost"):
    input_data = preprocess_input(project_type, num_workers, materials, project_length)
    prediction = model.predict(input_data)
    st.success(f"Estimated Project Cost: ${prediction[0]:,.2f}")
