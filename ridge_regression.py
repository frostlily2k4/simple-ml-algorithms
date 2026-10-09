
from sklearn.linear_model import Ridge
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

# Marks obtained
y = np.array([20, 30, 40, 50, 60, 70, 80, 90])

# Create the Ridge Regression model
model = Ridge(alpha=1.0)

# Train the model
model.fit(X, y)

# Predict marks for a student
prediction = model.predict([[5, 2]])

print("Predicted marks:", round(prediction[0], 2))
print("Model coefficients:", model.coef_)
