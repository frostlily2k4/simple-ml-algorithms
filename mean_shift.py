
# Algorithm #27: Mean Shift Clustering
# Repository: simple-ml-algorithms

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Sample dataset
# Features: annual spending score and purchase frequency
X = np.array([
    [10, 20], [12, 22], [11, 19],
    [14, 21], [9, 18],
    [50, 60], [52, 58], [48, 62],
    [55, 59], [49, 57],
    [90, 10], [92, 12], [88, 8],
    [94, 11], [86, 9]
])

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create the Mean Shift model
# Bandwidth controls the neighborhood size.
model = MeanShift(
    bandwidth=0.5,
    bin_seeding=True
)

# 4. Find clusters
labels = model.fit_predict(X_scaled)

# 5. Display cluster assignments
print("Algorithm: Mean Shift Clustering")
print("Number of clusters found:", len(np.unique(labels)))
print("\nData points and cluster labels:")

for point, label in zip(X, labels):
    print(f"Point {point} -> Cluster {label}")

# 6. Evaluate clustering quality
if len(np.unique(labels)) > 1:
    score = silhouette_score(X_scaled, labels)
    print("\nSilhouette Score:", round(score, 3))
else:
    print("\nOnly one cluster found; silhouette score unavailable.")

# 7. Display cluster centers in original feature units
centers = scaler.inverse_transform(model.cluster_centers_)

print("\nCluster centers:")
print(np.round(centers, 2))
