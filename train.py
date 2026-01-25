"""
train.py

This script:
1. Loads and cleans the Online Retail dataset
2. Performs RFM feature engineering
3. Trains a KMeans clustering model
4. Saves trained models for Streamlit usage
"""

# ========================
# Import required libraries
# ========================
import os
import pandas as pd
import numpy as np

from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import joblib


# ========================
# Configuration
# ========================
DATA_PATH = "data/online_retail.csv"
MODEL_DIR = "models"
N_CLUSTERS = 4
RANDOM_STATE = 42


# ========================
# Utility Functions
# ========================
def ensure_model_directory():
    """Create models directory if it does not exist."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


# ========================
# Data Loading & Cleaning
# ========================
def load_and_clean_data(path):
    """
    Load dataset and perform data cleaning.
    """
    df = pd.read_csv(path)

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled transactions
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid Quantity and UnitPrice
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Create TotalAmount feature
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df


# ========================
# RFM Feature Engineering
# ========================
def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features.
    """
    reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg({
            "InvoiceDate": lambda x: (reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalAmount": "sum"
        })
        .reset_index()
    )

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm


# ========================
# Model Training
# ========================
def train_kmeans(rfm):
    """
    Scale RFM features and train KMeans clustering model.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE
    )
    kmeans.fit(rfm_scaled)

    return scaler, kmeans


# ========================
# Product Recommendation Model
# ========================
def build_product_similarity(df):
    """
    Build item-based collaborative filtering similarity matrix.
    """
    pivot_table = (
        df.pivot_table(
            index="CustomerID",
            columns="Description",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
    )

    similarity_matrix = cosine_similarity(pivot_table.T)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=pivot_table.columns,
        columns=pivot_table.columns
    )

    return similarity_df


# ========================
# Main Execution
# ========================
def main():
    print("Starting training pipeline...")

    ensure_model_directory()

    # Load and clean data
    df_clean = load_and_clean_data(DATA_PATH)
    print("Data cleaned successfully.")

    # Create RFM features
    rfm = create_rfm_features(df_clean)
    print("RFM features created.")

    # Train clustering model
    scaler, kmeans = train_kmeans(rfm)
    print("KMeans model trained.")

    # Build product similarity matrix
    similarity_df = build_product_similarity(df_clean)
    print("Product similarity matrix created.")

    # Save models
    joblib.dump(scaler, f"{MODEL_DIR}/rfm_scaler.pkl")
    joblib.dump(kmeans, f"{MODEL_DIR}/rfm_kmeans.pkl")
    joblib.dump(similarity_df, f"{MODEL_DIR}/product_similarity.pkl")

    print("Models saved successfully in 'models/' directory.")
    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
