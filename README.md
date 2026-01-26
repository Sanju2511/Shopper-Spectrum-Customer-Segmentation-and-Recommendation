# 🛍️ Shopper Spectrum – Customer Segmentation using RFM & Clustering

## 📊 Project Overview
This project focuses on analyzing customer purchasing behavior using real-world
e-commerce transaction data. The primary goal is to segment customers into
meaningful groups that can help businesses design targeted marketing strategies,
improve customer retention, and understand customer value.

The project follows a complete data science workflow including data cleaning,
exploratory data analysis (EDA), feature engineering, unsupervised machine learning,
and business insight generation.

---

## 🎯 Business Problems Addressed
- How can customers be grouped based on their purchasing behavior?
- Which customers are high-value, regular, or at-risk?
- How can businesses use customer segments for data-driven decision-making?

---

## 🧠 Models Used
| Model | Purpose | Algorithm |
|------|--------|----------|
| Model 1 (Final) | Customer Segmentation | KMeans Clustering (RFM-based) |
| Model 2 | Cluster Validation | Hierarchical Clustering |
| Model 3 | Outlier Detection | DBSCAN |

> All models are unsupervised, making them suitable for real-world retail data
where labeled outcomes are unavailable.

---

## 📁 Dataset
The dataset contains transactional records from an online retail store, including:
- Invoice details
- Product information
- Quantity and price
- Customer ID
- Transaction timestamps
- Country

Due to GitHub file size limitations, the dataset is not included in this repository.
You can use the publicly available Online Retail Dataset for execution.

---

## 🔧 Key Techniques Used
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (Univariate, Bivariate & Multivariate)
- Feature Engineering using RFM Analysis
- Log Transformation & Standard Scaling
- Unsupervised Learning & Cluster Evaluation
- Business Insight Interpretation

---

## 📈 Key Insights
- Customer spending behavior is highly skewed, with a small group contributing most revenue
- RFM features effectively capture customer engagement and loyalty
- KMeans clustering produced well-separated and interpretable customer segments
- Hierarchical clustering validated the segmentation structure
- DBSCAN helped identify outlier customers with unusual purchasing patterns

---

## 🚀 How to Run the Project
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit application:
   streamlit run app.py

Ensure the dataset is available locally before running the app.

---

## 🛠️ Tools & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy, Joblib, Streamlit

---

## ✅ Conclusion
This project demonstrates how raw transactional data can be transformed into
actionable customer insights using unsupervised machine learning techniques.
The resulting customer segments can directly support personalized marketing,
customer retention strategies, and revenue optimization in e-commerce businesses.
