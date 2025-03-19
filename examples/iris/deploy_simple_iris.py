import os
import time
import subprocess
import tempfile
import argparse
import logging
from google.cloud import aiplatform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_simple_iris(project_id, region, endpoint_name=None, machine_type="n1-standard-2"):
    """Deploy a simple Iris classifier to Vertex AI."""
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Generate endpoint name if not provided
    if endpoint_name is None:
        timestamp = int(time.time())
        endpoint_name = f"iris-simple-{timestamp}"
    
    # Create a temporary directory for building
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Dockerfile path
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        
        # Copy the Dockerfile and server file to temp dir
        with open(dockerfile_path, "w") as f:
            with open("simple_iris_dockerfile", "r") as src:
                f.write(src.read())
        
        # Copy the server.py file
        server_path = os.path.join(temp_dir, "simple_iris_server.py")
        with open(server_path, "w") as f:
            with open("simple_iris_server.py", "r") as src:
                f.write(src.read())
        
        # Build and push the Docker image
        image_uri = f"{region}-docker.pkg.dev/{project_id}/mlflow-models/{endpoint_name}:{int(time.time())}"
        
        # Create cloud build file
        cloudbuild_yaml = os.path.join(temp_dir, "cloudbuild.yaml")
        with open(cloudbuild_yaml, "w") as f:
            f.write(f"""
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '{image_uri}', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', '{image_uri}']
images: ['{image_uri}']
""")
        
        # Run Cloud Build
        logger.info(f"Building and pushing container image to {image_uri}...")
        subprocess.run(
            [
                "gcloud", "builds", "submit", 
                "--project", project_id, 
                "--config", cloudbuild_yaml,
                temp_dir
            ],
            check=True
        )
        
        # Wait for the image to be available
        logger.info("Waiting for container image to be available...")
        time.sleep(10)
        
        # Create a model
        model = aiplatform.Model.upload(
            display_name=endpoint_name,
            serving_container_image_uri=image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            sync=True
        )
        logger.info(f"Model uploaded: {model.resource_name}")
        
        # Create an endpoint
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        logger.info(f"Endpoint created: {endpoint.resource_name}")
        
        # Deploy the model
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=1,
            sync=True
        )
        logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
        
        # Create a test script
        create_test_script(project_id, region, endpoint_name)
        
        return {
            "project_id": project_id,
            "region": region,
            "endpoint_name": endpoint_name,
            "endpoint_resource": endpoint.resource_name
        }

def create_test_script(project_id, region, endpoint_name):
    """Create a test script."""
    test_script = f"""
import os
from google.cloud import aiplatform

# Endpoint information
project_id = "{project_id}"
region = "{region}"
endpoint_name = "{endpoint_name}"

# Initialize Vertex AI
aiplatform.init(project=project_id, location=region)

# List endpoints to find the one with matching display name
endpoints = aiplatform.Endpoint.list(
    filter=f"display_name={endpoint_name}",
    order_by="create_time desc"
)

if not endpoints:
    raise ValueError(f"No endpoint found with display name: {{endpoint_name}}")

# Get the first (most recent) endpoint with matching display name
endpoint = endpoints[0]
print(f"Found endpoint: {{endpoint.resource_name}}")

# Test instances
test_instances = [
    [5.1, 3.5, 1.4, 0.2],  # Example of Iris-setosa
    [7.0, 3.2, 4.7, 1.4],  # Example of Iris-versicolor 
    [6.3, 3.3, 6.0, 2.5],  # Example of Iris-virginica
]

# Make a prediction
response = endpoint.predict(instances=test_instances)
predictions = response.predictions

# Map predictions to class names
class_names = ["setosa", "versicolor", "virginica"]
for i, prediction in enumerate(predictions):
    if isinstance(prediction, list):
        # Handle the case where prediction is a list of probabilities
        pred_class = prediction.index(max(prediction))
        print(f"Instance {{i+1}}: {{test_instances[i]}} -> Predicted: {{class_names[pred_class]}}")
    else:
        # Handle the case where prediction is a class index
        print(f"Instance {{i+1}}: {{test_instances[i]}} -> Predicted: {{class_names[int(prediction)]}}")
"""
    
    # Write the test script
    with open(f"test_{endpoint_name}.py", "w") as f:
        f.write(test_script)
    
    logger.info(f"Test script created: test_{endpoint_name}.py")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy a simple Iris classifier to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--endpoint-name", help="Custom name for the endpoint")
    parser.add_argument("--machine-type", default="n1-standard-2", help="Machine type for deployment")
    
    args = parser.parse_args()
    
    # Deploy the model
    deploy_info = deploy_simple_iris(
        project_id=args.project,
        region=args.region,
        endpoint_name=args.endpoint_name,
        machine_type=args.machine_type
    )
    
    # Print deployment info
    logger.info("\nDeployment completed successfully!")
    logger.info(f"Project ID: {deploy_info['project_id']}")
    logger.info(f"Region: {deploy_info['region']}")
    logger.info(f"Endpoint Name: {deploy_info['endpoint_name']}")
    logger.info(f"\nTo test your deployment, run: python test_{deploy_info['endpoint_name']}.py")

if __name__ == "__main__":
    main() 