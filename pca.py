from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
# [Height, Weight, Age]
X = np.array([
    [150, 50, 20],
    [160, 55, 22],
    [170, 65, 25],
    [180, 75, 28],
    [190, 85, 30]
])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create PCA model
pca = PCA(n_components=2)

# Reduce 3 features to 2 principal components
X_reduced = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

print("\nReduced data:")
print(X_reduced)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)