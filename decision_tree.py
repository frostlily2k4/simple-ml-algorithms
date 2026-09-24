from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Training data
# [Hours studied, Practice tests completed]
X = np.array([
    [1, 0],
    [2, 0],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 2],
    [7, 3],
    [8, 3]
])

# Labels
y = np.array([
    "Fail",
    "Fail",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
    "Pass"
])

# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Predict for a new student
prediction = model.predict([[5, 1]])

print("Predicted result:", prediction[0])