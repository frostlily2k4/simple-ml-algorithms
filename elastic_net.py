
# Algorithm #23: Elastic Net Regression
# Repository: simple-ml-algorithms

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Sample dataset
# Features: [house size in square feet, number of bedrooms]
X = np.array([
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4],
    [2200, 4],
    [2500, 5],
    [2800, 5],
    [3000, 6],
    [1600, 3],
    [2400, 4]
])

# Target: house price in thousands
y = np.array([
    80, 100, 120, 150, 180, 200,
    220, 250, 280, 300, 160, 240
])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Create the Elastic Net model
model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    max_iter=10000
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict prices for test data
y_pred = model.predict(X_test)

# 6. Display results
print("Algorithm: Elastic Net Regression")
print("Actual prices:", y_test)
print("Predicted prices:", np.round(y_pred, 2))

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 4))

# 8. Predict the price of a new house
new_house = np.array([[1700, 3]])
predicted_price = model.predict(new_house)

print("New house size: 1700 sq ft")
print("New house bedrooms: 3")
print(
    "Predicted house price:",
    round(predicted_price[0], 2),
    "thousand"
)
