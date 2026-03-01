# app.py
# ==========================================
# Shopper Spectrum – Customer Segmentation App
# ==========================================

from pathlib import Path

import streamlit as st
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SCALER_PATH = MODELS_DIR / "rfm_scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans_rfm_model.joblib"


@st.cache_resource
def load_artifacts():
    missing = [p for p in [SCALER_PATH, KMEANS_PATH] if not p.exists()]
    if missing:
        missing_text = "\n".join([f"- {p}" for p in missing])
        raise FileNotFoundError(
            "Required model artifacts are missing.\n"
            "Run the training notebook first and ensure these files exist:\n"
            f"{missing_text}"
        )

    scaler_local = joblib.load(SCALER_PATH)
    model_local = joblib.load(KMEANS_PATH)
    return scaler_local, model_local

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

try:
    scaler, kmeans_model = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.info(
        "Expected files:\n"
        "- models/rfm_scaler.joblib\n"
        "- models/kmeans_rfm_model.joblib"
    )
    st.stop()

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
