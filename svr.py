
# Algorithm #24: Support Vector Regression (SVR)
# Repository: simple-ml-algorithms

import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Sample dataset
# Feature: house size in square feet
X = np.array([
    [500], [700], [900], [1100],
    [1300], [1500], [1700], [1900],
    [2100], [2300], [2500], [2700]
])

# Target: house price in thousands
y = np.array([
    50, 70, 85, 105,
    125, 140, 165, 180,
    205, 220, 245, 260
])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Create an SVR model
# Scaling is important for SVR.
model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=100, epsilon=5)
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict house prices
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Algorithm: Support Vector Regression")
print("Actual prices:", y_test)
print("Predicted prices:", np.round(y_pred, 2))
print(
    "Mean Absolute Error:",
    round(mean_absolute_error(y_test, y_pred), 2)
)
print("R-squared Score:", round(r2_score(y_test, y_pred), 4))

# 7. Predict for a new house
new_house = np.array([[1600]])
predicted_price = model.predict(new_house)[0]

print("\nNew house size:", new_house[0, 0], "sq ft")
print("Predicted price:", round(predicted_price, 2), "thousand")
