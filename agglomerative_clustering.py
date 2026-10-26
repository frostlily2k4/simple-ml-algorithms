
# Algorithm #25: Agglomerative Clustering
# Repository: simple-ml-algorithms

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Sample dataset
# Features: annual spending and number of purchases
X = np.array([
    [10, 20],
    [12, 22],
    [11, 19],
    [50, 60],
    [52, 58],
    [48, 62],
    [90, 10],
    [92, 12],
    [88, 8]
])

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create the clustering model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

# 4. Fit the model and assign cluster labels
labels = model.fit_predict(X_scaled)

# 5. Display results
print("Algorithm: Agglomerative Clustering")
print("Number of clusters:", len(set(labels)))
print("\nData points and their cluster labels:")

for point, label in zip(X, labels):
    print(f"Point {point} -> Cluster {label}")

# 6. Evaluate clustering quality
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", round(score, 3))
print("(Closer to 1 generally indicates better-separated clusters.)")
