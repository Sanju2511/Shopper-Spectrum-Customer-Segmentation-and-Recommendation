"""
app.py

Streamlit application for:
1. Product Recommendation (Item-based Collaborative Filtering)
2. Customer Segmentation using RFM + KMeans
"""

# ========================
# Import required libraries
# ========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

st.markdown("---")


# ========================
# Load Trained Models
# ========================
@st.cache_resource
def load_models():
    """
    Load trained models and similarity matrix.
    """
    scaler = joblib.load("models/rfm_scaler.pkl")
    kmeans = joblib.load("models/rfm_kmeans.pkl")
    product_similarity = joblib.load("models/product_similarity.pkl")

    return scaler, kmeans, product_similarity


scaler, kmeans_model, similarity_df = load_models()


# ========================
# Sidebar Navigation
# ========================
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Module",
    ["Product Recommendation", "Customer Segmentation"]
)


# ======================================================
# MODULE 1: PRODUCT RECOMMENDATION
# ======================================================
if option == "Product Recommendation":

    st.header("🔍 Product Recommendation")

    st.write(
        "Enter a product name to get top 5 similar product recommendations "
        "based on customer purchase behavior."
    )

    # User input for product name
    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        if product_name in similarity_df.columns:

            # Get similarity scores for the selected product
            similarity_scores = (
                similarity_df[product_name]
                .sort_values(ascending=False)
                .iloc[1:6]
            )

            st.success("Top 5 Recommended Products:")
            for idx, product in enumerate(similarity_scores.index, start=1):
                st.write(f"{idx}. {product}")

        else:
            st.error("Product not found. Please check the product name.")


# ======================================================
# MODULE 2: CUSTOMER SEGMENTATION
# ======================================================
elif option == "Customer Segmentation":

    st.header("👤 Customer Segmentation")

    st.write(
        "Enter customer RFM values to predict the customer segment."
    )

    # User inputs for RFM values
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Customer Segment"):

        # Create DataFrame from user input
        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Predict cluster
        cluster = kmeans_model.predict(input_scaled)[0]

        # Map cluster to business labels
        cluster_labels = {
            0: "High-Value Customer",
            1: "Regular Customer",
            2: "At-Risk Customer",
            3: "Occasional Customer"
        }

        st.success(f"Predicted Segment: {cluster_labels.get(cluster)}")
