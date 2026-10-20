
from sklearn.ensemble import ExtraTreesClassifier
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
    [8, 3],
    [2, 1],
    [7, 2]
])

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])

# Create the Extra Trees model
model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict for a new student
new_student = np.array([[5, 2]])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Predicted result: Pass")
else:
    print("Predicted result: Fail")

print("Class probabilities:", model.predict_proba(new_student))
