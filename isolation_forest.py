
from sklearn.ensemble import IsolationForest
import numpy as np

# Sample data: [hours studied, practice tests completed]
X = np.array([
    [2, 1],
    [2.2, 1.1],
    [1.8, 0.9],
    [2.1, 1.2],
    [2.3, 1.0],
    [1.9, 1.1],
    [2.0, 0.8],
    [2.2, 0.9],
    [10, 10]  # Unusual data point
])

# Create the Isolation Forest model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

# Train the model and detect anomalies
predictions = model.fit_predict(X)

print("Anomaly labels:")
print(predictions)

# Count normal points and anomalies
normal_count = np.sum(predictions == 1)
anomaly_count = np.sum(predictions == -1)

print("\nNumber of normal points:", normal_count)
print("Number of anomalies:", anomaly_count)

# Display detected anomalies
print("\nDetected anomalies:")
print(X[predictions == -1])
