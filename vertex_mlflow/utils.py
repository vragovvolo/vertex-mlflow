import os
import tempfile
import uuid
import logging
from google.cloud.devtools import cloudbuild_v1

logger = logging.getLogger(__name__)

def generate_dockerfile(model_uri):
    """
    Generates a Dockerfile for serving an MLflow model.
    
    The Dockerfile will use a base Python image, install MLflow,
    and run `mlflow models serve` on the model.
    
    Returns the path to the generated Dockerfile.
    """
    dockerfile_content = f"""
FROM python:3.9-slim
RUN pip install --no-cache-dir mlflow==2.7.0
# Copy the MLflow model artifacts (assuming they are in /app/model)
COPY . /app
WORKDIR /app
EXPOSE 8080
# Start the MLflow model server
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080"]
"""
    temp_dir = tempfile.mkdtemp()
    dockerfile_path = os.path.join(temp_dir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    logger.info("Dockerfile generated at: %s", dockerfile_path)
    return dockerfile_path

def build_docker_image(project, region, dockerfile_path, artifact_registry_repo, image_tag):
    """
    Builds a Docker image using Cloud Build and pushes it to Artifact Registry.
    
    Parameters:
      - project: GCP project ID.
      - region: GCP region.
      - dockerfile_path: Path to the Dockerfile.
      - artifact_registry_repo: Artifact Registry repo (e.g., us-central1-docker.pkg.dev/my-project/my-repo)
      - image_tag: Tag to use for the image.
      
    Returns:
      The full image URI.
    """
    client = cloudbuild_v1.services.cloud_build.CloudBuildClient()
    build = {
        "steps": [{
            "name": "gcr.io/cloud-builders/docker",
            "args": [
                "build",
                "-t", f"{artifact_registry_repo}/{image_tag}:latest",
                "."
            ]
        }],
        "images": [f"{artifact_registry_repo}/{image_tag}:latest"]
    }
    
    # Upload the build context (the folder with the Dockerfile) to Cloud Build.
    # For simplicity, we assume the Dockerfile is in the context root.
    # In production, you would tar the context and pass it to Cloud Build.
    logger.info("Starting Cloud Build for image: %s/%s:latest", artifact_registry_repo, image_tag)
    build_operation = client.create_build(project_id=project, build=build)
    result = build_operation.result()  # Wait for build to finish.
    
    image_uri = f"{artifact_registry_repo}/{image_tag}:latest"
    logger.info("Docker image built and pushed: %s", image_uri)
    return image_uri
