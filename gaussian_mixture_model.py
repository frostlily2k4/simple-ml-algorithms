
# Algorithm #26: Gaussian Mixture Model (GMM)
# Repository: simple-ml-algorithms

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Create sample data
# Features: annual spending and purchase frequency
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

# 3. Create the GMM model
model = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42,
    n_init=5
)

# 4. Train the model
model.fit(X_scaled)

# 5. Predict cluster labels
labels = model.predict(X_scaled)

# 6. Calculate membership probabilities
probabilities = model.predict_proba(X_scaled)

# 7. Display results
print("Algorithm: Gaussian Mixture Model")
print("Number of clusters:", model.n_components)
print("\nCluster assignments:")

for point, label in zip(X, labels):
    print(f"Data point {point} -> Cluster {label}")

print("\nMembership probabilities for first 5 points:")
print(np.round(probabilities[:5], 3))

# 8. Evaluate clustering quality
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", round(score, 3))

# 9. Predict the cluster for a new data point
new_point = np.array([[51, 59]])
new_scaled = scaler.transform(new_point)

new_cluster = model.predict(new_scaled)
new_probabilities = model.predict_proba(new_scaled)

print("\nNew data point:", new_point[0])
print("Predicted cluster:", new_cluster[0])
print("Membership probabilities:",
      np.round(new_probabilities[0], 3))
