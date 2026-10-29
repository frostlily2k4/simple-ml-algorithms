
# Algorithm #28: OPTICS Clustering
# Repository: simple-ml-algorithms

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Create a sample dataset
# Features: spending score and purchase frequency
X = np.array([
    [10, 10], [11, 11], [10, 12], [12, 10],
    [13, 12], [11, 13],

    [50, 50], [51, 51], [52, 50], [50, 52],
    [53, 51], [51, 53],

    [90, 10], [91, 11], [92, 10], [90, 12],
    [93, 11], [91, 13],

    [40, 90], [100, 100]
])

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create the OPTICS model
model = OPTICS(
    min_samples=3,
    max_eps=np.inf,
    cluster_method="xi",
    xi=0.05
)

# 4. Fit the model and predict cluster labels
labels = model.fit_predict(X_scaled)

# 5. Display results
print("Algorithm: OPTICS Clustering")
print("Number of clusters found:",
      len(set(labels) - {-1}))

print("\nData points and cluster labels:")
for point, label in zip(X, labels):
    if label == -1:
        print(f"Point {point} -> Noise / Outlier")
    else:
        print(f"Point {point} -> Cluster {label}")

# 6. Count noise points
noise_count = np.sum(labels == -1)
print("\nNumber of noise points:", noise_count)

# 7. Evaluate clustering, excluding noise
cluster_labels = labels[labels != -1]
cluster_points = X_scaled[labels != -1]

if len(set(cluster_labels)) >= 2:
    score = silhouette_score(cluster_points, cluster_labels)
    print("Silhouette Score (excluding noise):",
          round(score, 3))
else:
    print("Silhouette Score unavailable: "
          "at least two non-noise clusters are needed.")

print("\nOPTICS execution completed!")
