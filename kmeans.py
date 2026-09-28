from sklearn.cluster import KMeans
import numpy as np

# Customer data
# [Annual spending, Number of purchases]
X = np.array([
    [10, 2],
    [12, 3],
    [15, 2],
    [50, 8],
    [52, 9],
    [55, 8],
    [90, 15],
    [92, 16],
    [95, 15]
])

# Create the K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
model.fit(X)

# Get cluster labels
labels = model.labels_

print("Cluster labels:")
print(labels)

print("\nCluster centers:")
print(model.cluster_centers_)

# Predict the cluster for a new customer
new_customer = np.array([[53, 8]])
prediction = model.predict(new_customer)

print("\nNew customer belongs to cluster:", prediction[0])