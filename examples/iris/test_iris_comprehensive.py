import os
from google.cloud import aiplatform

# Endpoint information
project_id = "protean-bit-234523"
region = "northamerica-northeast1"
endpoint_name = "iris-simple"

# Initialize Vertex AI
aiplatform.init(project=project_id, location=region)

# List endpoints to find the one with matching display name
endpoints = aiplatform.Endpoint.list(
    filter=f"display_name={endpoint_name}",
    order_by="create_time desc"
)

if not endpoints:
    raise ValueError(f"No endpoint found with display name: {endpoint_name}")

# Get the first (most recent) endpoint with matching display name
endpoint = endpoints[0]
print(f"Found endpoint: {endpoint.resource_name}")

# More comprehensive test instances
test_instances = [
    # Setosa examples (features: small petals, medium sepals)
    [5.1, 3.5, 1.4, 0.2],  # Classic setosa
    [4.9, 3.0, 1.4, 0.2],  # Another setosa
    [4.7, 3.2, 1.3, 0.2],  # Yet another setosa
    [5.0, 3.6, 1.4, 0.2],  # Setosa with larger sepal width
    [5.4, 3.9, 1.7, 0.4],  # Setosa with larger dimensions overall
    
    # Versicolor examples (features: medium petals, medium sepals)
    [7.0, 3.2, 4.7, 1.4],  # Classic versicolor
    [6.4, 3.2, 4.5, 1.5],  # Another versicolor
    [6.9, 3.1, 4.9, 1.5],  # Yet another versicolor
    [5.6, 2.7, 4.2, 1.3],  # Versicolor with smaller sepals
    [6.7, 3.1, 4.4, 1.4],  # Slightly different versicolor
    
    # Virginica examples (features: large petals, larger sepals)
    [6.3, 3.3, 6.0, 2.5],  # Classic virginica
    [5.8, 2.7, 5.1, 1.9],  # Another virginica
    [7.1, 3.0, 5.9, 2.1],  # Yet another virginica
    [6.5, 3.0, 5.8, 2.2],  # Slightly different virginica
    [7.6, 3.0, 6.6, 2.1],  # Virginica with larger dimensions
    
    # Borderline/challenging cases
    [6.0, 2.2, 5.0, 1.5],  # Could be versicolor or virginica
    [5.7, 2.8, 4.5, 1.3],  # Could be versicolor with smaller petals
    [6.7, 3.3, 5.7, 2.1],  # Could be virginica with slightly smaller petals
    [6.4, 2.8, 5.6, 2.2],  # Virginica with different proportions
    [5.5, 2.3, 4.0, 1.3]   # Versicolor with different proportions
]

# Expected classes
expected_classes = [
    0, 0, 0, 0, 0,   # Setosa (class 0)
    1, 1, 1, 1, 1,   # Versicolor (class 1)
    2, 2, 2, 2, 2,   # Virginica (class 2)
    1, 1, 2, 2, 1    # Borderline cases (expected classifications)
]

# Make a prediction
print(f"Sending {len(test_instances)} test instances for prediction...\n")
response = endpoint.predict(instances=test_instances)
predictions = response.predictions

# Map predictions to class names and check accuracy
class_names = ["setosa", "versicolor", "virginica"]
correct = 0

print("PREDICTION RESULTS:")
print("-" * 70)
print(f"{'#':<3} {'SEPAL LEN':<10} {'SEPAL WID':<10} {'PETAL LEN':<10} {'PETAL WID':<10} {'PREDICTED':<12} {'EXPECTED':<12} {'CORRECT':<8}")
print("-" * 70)

for i, (features, prediction, expected) in enumerate(zip(test_instances, predictions, expected_classes)):
    # Extract features
    sepal_length, sepal_width, petal_length, petal_width = features
    
    # Get class prediction (handle both list and scalar outputs)
    if isinstance(prediction, list):
        pred_class = prediction.index(max(prediction))
    else:
        pred_class = int(prediction)
    
    is_correct = pred_class == expected
    if is_correct:
        correct += 1
    
    print(f"{i+1:<3} {sepal_length:<10.1f} {sepal_width:<10.1f} {petal_length:<10.1f} {petal_width:<10.1f} {class_names[pred_class]:<12} {class_names[expected]:<12} {'✓' if is_correct else '✗'}")

print("-" * 70)
accuracy = 100 * correct / len(test_instances)
print(f"Accuracy: {correct}/{len(test_instances)} = {accuracy:.1f}%")

print("\nPrediction Statistics:")
print(f"Setosa: {predictions.count(0) if isinstance(predictions[0], int) else sum(1 for p in predictions if p == 0)}")
print(f"Versicolor: {predictions.count(1) if isinstance(predictions[0], int) else sum(1 for p in predictions if p == 1)}")
print(f"Virginica: {predictions.count(2) if isinstance(predictions[0], int) else sum(1 for p in predictions if p == 2)}") 