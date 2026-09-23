from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Training data
X = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [8, 8],
    [9, 9],
    [10, 10]
])

# Labels
y = np.array([
    "Class A",
    "Class A",
    "Class A",
    "Class B",
    "Class B",
    "Class B"
])

# Create the KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Predict the class of a new data point
prediction = model.predict([[4, 4]])

print("Predicted class:", prediction[0])