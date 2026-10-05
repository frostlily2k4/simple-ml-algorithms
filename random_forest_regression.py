from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Training data
# [Hours studied]
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

# Marks obtained
y = np.array([
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90
])

# Create the Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict marks for a student who studied 5 hours
prediction = model.predict([[5]])

print("Predicted marks:", prediction[0])