from sklearn.cluster import DBSCAN
import numpy as np

# Sample data
# Two groups of nearby points and one possible outlier
X = np.array([
    [1, 1],
    [1.2, 1.1],
    [0.8, 1.2],
    [1.1, 0.9],
    [5, 5],
    [5.2, 5.1],
    [4.8, 5.2],
    [5.1, 4.9],
    [10, 10]
])

# Create the DBSCAN model
model = DBSCAN(eps=0.5, min_samples=3)

# Find clusters
labels = model.fit_predict(X)

print("Cluster labels:")
print(labels)

# Number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Number of noise points
n_noise = list(labels).count(-1)

print("\nNumber of clusters:", n_clusters)
print("Number of noise points:", n_noise)