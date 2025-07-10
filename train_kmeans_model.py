import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load the dataset (make sure this file exists)
df = pd.read_excel("customers_with_clusters.xlsx")

# Select features used for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save the trained model
joblib.dump(kmeans, "kmeans_model.pkl")

print("✅ Model trained and saved as 'kmeans_model.pkl'")
