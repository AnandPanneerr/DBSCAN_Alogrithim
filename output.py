import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load pickle files
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("X_pca.pkl", "rb") as f:
    X_pca = pickle.load(f)

with open("dbscan_labels.pkl", "rb") as f:
    labels = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍷 Wine Cluster Prediction (DBSCAN)")

st.subheader("Enter New Wine Chemical Values")

alcohol = st.number_input("Alcohol", value=13.0)
malic_acid = st.number_input("Malic Acid", value=2.0)
ash = st.number_input("Ash", value=2.3)
ash_alcanity = st.number_input("Ash Alcanity", value=19.0)
magnesium = st.number_input("Magnesium", value=100)
total_phenols = st.number_input("Total Phenols", value=2.5)
flavanoids = st.number_input("Flavanoids", value=2.5)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.3)
proanthocyanins = st.number_input("Proanthocyanins", value=1.6)
color_intensity = st.number_input("Color Intensity", value=5.0)
hue = st.number_input("Hue", value=1.0)
od280 = st.number_input("OD280", value=3.0)

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("Predict Cluster"):
    new_sample = np.array([[ 
        alcohol, malic_acid, ash, ash_alcanity, magnesium,
        total_phenols, flavanoids, nonflavanoid_phenols,
        proanthocyanins, color_intensity, hue, od280
    ]])

    # Scale → PCA
    new_scaled = scaler.transform(new_sample)
    new_pca = pca.transform(new_scaled)

    # Nearest-point cluster assignment (DBSCAN workaround)
    distances = np.linalg.norm(X_pca - new_pca, axis=1)
    nearest_idx = np.argmin(distances)
    predicted_cluster = labels[nearest_idx]

    st.success(f"✅ Predicted Cluster: {predicted_cluster}")

    # -----------------------------
    # PCA Visualization
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.5)
    ax.scatter(
        new_pca[0, 0], new_pca[0, 1],
        color="red", s=120, edgecolor="black"
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("DBSCAN Clusters (PCA View)")

    st.pyplot(fig)
