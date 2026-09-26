from sklearn.naive_bayes import GaussianNB
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
# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Create the Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X, y)

# Predict for a new student
prediction = model.predict([[5, 1]])

if prediction[0] == 1:
    print("Predicted result: Pass")
else:
    print("Predicted result: Fail")