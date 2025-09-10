import streamlit as st 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model, scaler, and dataset
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("customers_with_clusters.csv")  # already has Cluster column

# Cluster descriptions (must match your training file)
cluster_desc = {
    0: "Older, Moderate Spenders",
    1: "Young, High Spenders",
    2: "Very Young, Average Spenders",
    3: "Mid-aged, Low Spenders"
}

st.title("Customer Segmentation with KMeans")

# -------------------------
# User inputs
# -------------------------
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
spending = st.number_input("Enter Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Predict cluster for this input
features = np.array([[age, spending]])
features_scaled = scaler.transform(features)
cluster = model.predict(features_scaled)[0]

st.write(f"🟢 This customer belongs to **Cluster {cluster}**")
st.write(f"💡 Cluster Meaning: {cluster_desc[cluster]}")

# -------------------------
# Checkbox: Dataset & Plot
# -------------------------
if st.checkbox("Show Dataset & Cluster Visualization"):
    st.subheader("Customer Dataset with Cluster Labels")
    st.dataframe(df.head(20))  # show first 20 rows

    st.subheader("Cluster Scatter Plot")
    plt.figure(figsize=(8,5))
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        cluster_points = df[df['Cluster'] == i]
        plt.scatter(cluster_points['Age'], cluster_points['Spending Score (1-100)'], 
                    color=colors[i], label=f"Cluster {i}: {cluster_desc[i]}")
    # New customer
    plt.scatter(age, spending, color='black', s=120, label="New Customer", marker='X')
    plt.xlabel("Age")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments (K-Means)")
    plt.legend()
    st.pyplot(plt)
