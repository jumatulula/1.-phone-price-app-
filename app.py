import streamlit as st
import numpy as np
import joblib
import pandas as pd

# 1. Load the trained model and scaler locally
model = joblib.load('used_phone_price_model.pkl')
scaler = joblib.load('data_scaler.pkl')

# 2. Design the webpage UI layout
st.set_page_config(page_title="Used Phone Price Predictor", page_icon="📱", layout="centered")

st.title("📱 Used Phones & Tablets Price Predictor")
st.write("Fill in the device specifications below, and our machine learning AI will predict its market price instantly.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    ram = st.number_input("RAM Capacity (GB)", min_value=1, max_value=64, value=4, step=1)
    battery = st.number_input("Battery Capacity (mAh)", min_value=500, max_value=10000, value=4000, step=100)
    screen_size = st.number_input("Screen Size (Inches)", min_value=3.0, max_value=15.0, value=6.1, step=0.1)

with col2:
    rear_camera = st.number_input("Rear Camera Resolution (MP)", min_value=2, max_value=200, value=12, step=1)
    front_camera = st.number_input("Front Camera Resolution (MP)", min_value=1, max_value=100, value=8, step=1)
    internal_memory = st.number_input("Internal Memory (GB)", min_value=4, max_value=1024, value=64, step=4)

st.markdown("### Estimate Price")
if st.button("Predict Best Market Price", use_container_width=True):
    # Prepare input dictionary with exact column names your scaler expects
    input_data = {
        'ram': ram,
        'battery': battery,
        'screen_size': screen_size,
        'rear_camera_mp': rear_camera,
        'front_camera_mp': front_camera,
        'internal_memory': internal_memory
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        # EXTRACT COLUMNS DIRECTLY FROM THE SCALER (NO COLAB REQUIRED!)
        feature_names = scaler.feature_names_in_
        
        # Fill missing one-hot encoded columns (like phone brands) with 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match the trained data exactly
        input_df = input_df[feature_names]
        
        # Preprocess, Scale, and Predict
        scaled_input = scaler.transform(input_df)
        raw_pred = model.predict(scaled_input)
        
        # Reverse log transform to get actual currency price
        final_market_price = np.exp(raw_pred)
        
        st.success(f"💰 **Estimated Used Market Value: ${final_market_price[0]:.2f}**")
        st.info("Note: This estimation assumes standard phone physical condition without visible screen cracks.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
