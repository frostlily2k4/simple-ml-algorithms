from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
prediction = model.predict([[6]])

print("Predicted value for 6:", prediction[0])