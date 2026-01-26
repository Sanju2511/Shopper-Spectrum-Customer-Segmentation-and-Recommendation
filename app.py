# app.py
# ==========================================
# Shopper Spectrum – Customer Segmentation App
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------
# Load trained models and scaler
# ------------------------------------------
scaler = joblib.load("rfm_scaler.joblib")
kmeans_model = joblib.load("kmeans_rfm_model.joblib")

# ------------------------------------------
# App Title and Description
# ------------------------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛍️ Shopper Spectrum")
st.subheader("Customer Segmentation using RFM & KMeans Clustering")

st.write(
    """
This application predicts the **customer segment** based on  
**Recency, Frequency, and Monetary (RFM)** values.

The segmentation helps businesses identify:
- High-value customers
- Regular customers
- Occasional buyers
- At-risk customers
"""
)

# ------------------------------------------
# User Inputs
# ------------------------------------------
st.header("🔢 Enter Customer RFM Values")

recency = st.number_input(
    "Recency (Days since last purchase)",
    min_value=0,
    max_value=2000,
    value=30
)

frequency = st.number_input(
    "Frequency (Number of purchases)",
    min_value=1,
    max_value=1000,
    value=5
)

monetary = st.number_input(
    "Monetary Value (Total spend)",
    min_value=1.0,
    max_value=1000000.0,
    value=500.0
)

# ------------------------------------------
# Prediction Logic
# ------------------------------------------
def predict_customer_segment(recency, frequency, monetary):
    # Log transformation to reduce skew
    r_log = np.log1p(recency)
    f_log = np.log1p(frequency)
    m_log = np.log1p(monetary)

    # Scale features
    scaled_features = scaler.transform([[r_log, f_log, m_log]])

    # Predict cluster
    cluster = kmeans_model.predict(scaled_features)[0]
    return cluster

# ------------------------------------------
# Predict Button
# ------------------------------------------
if st.button("🔍 Predict Customer Segment"):
    cluster = predict_customer_segment(recency, frequency, monetary)

    st.success(f"✅ Predicted Customer Segment: **Cluster {cluster}**")

    # Optional interpretation
    st.markdown("### 📌 Segment Interpretation")
    if cluster == 0:
        st.write("🟢 **High-Value Customer** – Frequent and high spending.")
    elif cluster == 1:
        st.write("🟡 **Regular Customer** – Consistent but moderate spending.")
    elif cluster == 2:
        st.write("🔵 **Occasional Customer** – Infrequent purchases.")
    else:
        st.write("🔴 **At-Risk Customer** – Low engagement, needs retention strategies.")

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Built using RFM Analysis, KMeans Clustering & Streamlit")
