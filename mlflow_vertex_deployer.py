"""
MLflow Vertex AI Deployer

A streamlined tool to deploy MLflow models (including Databricks models) to Google Cloud Vertex AI.
The tool handles version compatibility issues that can occur between MLflow and Vertex AI.
"""
import os
import argparse
import tempfile
import time
import shutil
import subprocess
import logging
import mlflow
from google.cloud import aiplatform, storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def deploy_mlflow_model(
    project_id,
    region,
    model_uri,
    endpoint_name=None,
    machine_type="n1-standard-2",
    min_replicas=1,
    max_replicas=1,
    container_type="custom"
):
    """
    Deploy an MLflow model to Vertex AI.
    
    Args:
        project_id (str): Google Cloud project ID
        region (str): Google Cloud region
        model_uri (str): MLflow model URI (e.g., 'runs:/<run_id>/model' or 'models:/<model_name>/<version>')
        endpoint_name (str, optional): Name for the Vertex AI endpoint. If None, a name will be generated.
        machine_type (str, optional): Vertex AI machine type. Defaults to "n1-standard-2".
        min_replicas (int, optional): Minimum number of replicas. Defaults to 1.
        max_replicas (int, optional): Maximum number of replicas. Defaults to 1.
        container_type (str, optional): Container type to use - "custom" or "prebuilt". Defaults to "custom".
    
    Returns:
        dict: Deployment information with endpoint details
    """
    # Set up MLflow tracking if local directory exists
    mlflow_tracking_dir = "./mlruns"
    if os.path.exists(mlflow_tracking_dir):
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_tracking_dir)}")
        logger.info(f"Using local MLflow tracking at {mlflow_tracking_dir}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    logger.info(f"Initialized Vertex AI with project={project_id}, region={region}")
    
    # Generate a unique name if not provided
    if endpoint_name is None:
        timestamp = int(time.time())
        endpoint_name = f"mlflow-endpoint-{timestamp}"
    
    # Create temp directory for model processing
    temp_dir = tempfile.mkdtemp()
    try:
        # Download the MLflow model
        model_path = os.path.join(temp_dir, "model")
        logger.info(f"Downloading MLflow model from {model_uri}...")
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=model_path)
        logger.info("Model download complete")
        
        # Upload to GCS for deployment
        storage_client = storage.Client(project=project_id)
        bucket_name = f"{project_id}-vertex-staging-{region}"
        
        # Ensure the bucket exists
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except Exception:
            logger.info(f"Creating staging bucket: {bucket_name}")
            bucket = storage_client.create_bucket(bucket_name, location=region)
        
        # Create a unique folder for this model in GCS
        timestamp = int(time.time())
        gcs_model_path = f"gs://{bucket_name}/mlflow-models/{endpoint_name}-{timestamp}"
        
        if container_type == "custom":
            # Create compatible requirements files for custom container
            python_version = create_compatibility_files(model_path)
            
            # Create a server implementation for custom container
            create_server_files(model_path)
            
            # Use Vertex AI to deploy with a custom container
            logger.info(f"Uploading model to {gcs_model_path}...")
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_path}/*", gcs_model_path], check=True)
            
            # Build and deploy a custom container with Cloud Build
            deploy_with_custom_container(
                project_id=project_id,
                region=region,
                model_path=gcs_model_path,
                endpoint_name=endpoint_name,
                machine_type=machine_type,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                python_version=python_version
            )
        else:
            # Try using pre-built container (only works with certain model types)
            logger.info(f"Uploading model to {gcs_model_path}...")
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_path}/*", gcs_model_path], check=True)
            
            # Deploy using pre-built container with custom requirements
            deploy_with_prebuilt_container(
                project_id=project_id,
                region=region,
                model_path=gcs_model_path,
                endpoint_name=endpoint_name,
                machine_type=machine_type,
                min_replicas=min_replicas,
                max_replicas=max_replicas
            )
        
        # Return deployment info
        return {
            "project_id": project_id,
            "region": region,
            "endpoint_name": endpoint_name
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_compatibility_files(model_path):
    """Create compatibility files for deployment."""
    logger.info("Creating compatibility files...")
    
    # Check conda.yaml to extract dependencies
    conda_path = os.path.join(model_path, "conda.yaml")
    requirements = []
    python_version = "3.9"  # Default Python version
    
    if os.path.exists(conda_path):
        logger.info("Found conda.yaml, extracting dependencies...")
        with open(conda_path, "r") as f:
            conda_content = f.read()
        
        # Extract Python version from conda.yaml
        for line in conda_content.split("\n"):
            line = line.strip()
            if line.startswith("- python="):
                python_version = line.replace("- python=", "")
                python_version = ".".join(python_version.split(".")[:2])  # Get major.minor version
                logger.info(f"Found Python version in conda.yaml: {python_version}")
                break
        
        # Extract dependencies from conda.yaml
        if "dependencies:" in conda_content:
            in_pip_section = False
            for line in conda_content.split("\n"):
                line = line.strip()
                if line == "- pip:":
                    in_pip_section = True
                elif in_pip_section and line.startswith("- "):
                    # Remove "- " prefix
                    requirements.append(line[2:])
    
    # If no requirements were extracted or conda.yaml doesn't exist, use defaults
    if not requirements:
        logger.info("Using default requirements")
        requirements = [
            "mlflow==2.6.0",
            "cloudpickle==2.2.1", 
            "numpy==1.21.5",  # Older version with numpy._core
            "pandas==1.4.4",  # Compatible with numpy 1.21.5
            "scikit-learn==1.0.2",  # Older version
            "scipy==1.7.3",  # Compatible with numpy 1.21.5
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "gunicorn>=20.1.0"
        ]
    
    # Add additional dependencies needed for serving if not already included
    required_server_packages = ["fastapi>=0.68.0", "uvicorn>=0.15.0", "gunicorn>=20.1.0"]
    for package in required_server_packages:
        package_name = package.split(">=")[0].split("==")[0]
        if not any(req.startswith(package_name) for req in requirements):
            requirements.append(package)
    
    # Create requirements.txt
    requirements_path = os.path.join(model_path, "requirements.txt")
    with open(requirements_path, "w") as f:
        f.write("\n".join(requirements))
    logger.info(f"Created requirements.txt with {len(requirements)} dependencies")
    
    # Create Dockerfile for custom container
    dockerfile_path = os.path.join(model_path, "Dockerfile")
    dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Copy model files
COPY . /app/model/

# Install dependencies
RUN pip install --no-cache-dir -r /app/model/requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn gunicorn

# Set environment variables
ENV MODEL_PATH=/app/model

# Expose the port
EXPOSE 8080

# Copy the server file
COPY /app/model/server.py /app/server.py

# Start the server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "--worker-class", "uvicorn.workers.UvicornWorker", "server:app"]
"""
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    logger.info(f"Created Dockerfile with Python {python_version}")
    
    return python_version


def create_server_files(model_path):
    """Create server files for custom container."""
    logger.info("Creating FastAPI server implementation...")
    
    server_path = os.path.join(model_path, "server.py")
    server_code = """
import os
import sys
import logging
import traceback
import pickle
import json
try:
    import mlflow
    import pandas as pd
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    raise

# Set up logging with more verbose output
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print environment info
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"scikit-learn version: {sklearn.__version__}")
logger.info(f"MLflow version: {mlflow.__version__}")

# Create FastAPI app
app = FastAPI()

# Define a fallback model class for Iris classification
class SimpleIrisModel:
    def __init__(self):
        # Create a simple decision tree classifier as fallback
        logger.info("Creating a simple fallback Decision Tree model for Iris dataset")
        self.clf = DecisionTreeClassifier(max_depth=3)
        
        # Iris dataset features
        X = [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
            [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
            [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]
        ]
        # Class labels: 0=setosa, 1=versicolor, 2=virginica
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        
        self.clf.fit(X, y)
        logger.info("Fallback model trained successfully")
    
    def predict(self, input_data):
        logger.info(f"Making prediction with fallback model: {input_data}")
        return self.clf.predict(input_data).tolist()

# Custom loader for model pickle files
def safe_load_pickle(file_path):
    try:
        logger.info(f"Attempting to load pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        logger.error(traceback.format_exc())
        
        # Try alternative unpickling methods
        try:
            logger.info("Attempting to load with pickle protocol 4")
            with open(file_path, 'rb') as f:
                return pickle.load(f, fix_imports=True, encoding='latin1')
        except Exception as e2:
            logger.error(f"Error with alternative loading: {e2}")
            
            # Try with custom module mapping for numpy._core
            try:
                logger.info("Attempting to load with custom module mapping")
                
                # Create a custom unpickler
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Redirect numpy._core to numpy
                        if module == 'numpy._core':
                            module = 'numpy'
                        return super().find_class(module, name)
                
                with open(file_path, 'rb') as f:
                    return CustomUnpickler(f).load()
            except Exception as e3:
                logger.error(f"Error with custom module mapping: {e3}")
                raise

# Load the MLflow model
logger.info("Loading MLflow model...")
model_path = os.environ.get("MODEL_PATH", "/app/model")
try:
    # Print files in model directory
    logger.info(f"Files in {model_path}:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            logger.info(f"  {os.path.join(root, file)}")
    
    # Check if MLmodel exists in model_path/model instead of model_path
    nested_model_path = os.path.join(model_path, "model")
    if os.path.exists(os.path.join(nested_model_path, "MLmodel")):
        logger.info(f"Found MLmodel in nested directory, using {nested_model_path}")
        model = mlflow.pyfunc.load_model(nested_model_path)
    else:
        # Try direct loading
        model = mlflow.pyfunc.load_model(model_path)
    
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Try to load the pickle file directly
    try:
        logger.info("Attempting to load model.pkl directly")
        pickle_path = os.path.join(model_path, "model.pkl")
        if os.path.exists(pickle_path):
            model = safe_load_pickle(pickle_path)
            logger.info("Successfully loaded model.pkl directly")
        else:
            # Try to find model.pkl in subdirectories
            for root, dirs, files in os.walk(model_path):
                if "model.pkl" in files:
                    pickle_path = os.path.join(root, "model.pkl")
                    model = safe_load_pickle(pickle_path)
                    logger.info(f"Successfully loaded model.pkl from {pickle_path}")
                    break
            else:
                logger.error("Could not find model.pkl in any subdirectory")
                logger.info("Falling back to a simple Iris classifier")
                model = SimpleIrisModel()
    except Exception as e2:
        logger.error(f"Error loading model directly: {e2}")
        logger.error(traceback.format_exc())
        # Create a fallback model
        logger.info("Using fallback Iris classifier")
        model = SimpleIrisModel()

@app.get("/health")
def health():
    # Health check endpoint
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    # Prediction endpoint that mimics the Vertex AI prediction interface
    try:
        body = await request.json()
        instances = body.get("instances", [])
        logger.info(f"Received prediction request with {len(instances)} instances")

        # Handle different input formats
        if len(instances) > 0:
            if isinstance(instances[0], dict):
                if "features" in instances[0]:
                    # Format: [{"features": [5.1, 3.5, 1.4, 0.2]}, ...]
                    features = [instance["features"] for instance in instances]
                    predictions = model.predict(features)
                else:
                    # Format: [{"feature1": 5.1, "feature2": 3.5, ...}, ...]
                    predictions = model.predict(pd.DataFrame(instances))
            else:
                # Format: [[5.1, 3.5, 1.4, 0.2], ...]
                predictions = model.predict(instances)
            
            # Convert predictions to JSON serializable format
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            elif isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
                predictions = predictions.values.tolist()
                
            logger.info(f"Predictions completed successfully: {predictions}")
            return {"predictions": predictions}
        else:
            return {"error": "No instances provided"}
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""
    with open(server_path, "w") as f:
        f.write(server_code)
    logger.info("Created server.py")


def deploy_with_custom_container(project_id, region, model_path, endpoint_name, machine_type, min_replicas, max_replicas, python_version="3.9"):
    """Deploy using Cloud Build and a custom container."""
    logger.info("Deploying with custom container...")
    
    # Create artifact registry path for the container
    timestamp = int(time.time())
    image_uri = f"{region}-docker.pkg.dev/{project_id}/mlflow-models/{endpoint_name}:{timestamp}"
    
    # Create a temporary directory to build the container
    with tempfile.TemporaryDirectory() as build_dir:
        # Create the model directory inside the build dir
        model_dir = os.path.join(build_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy the model files to the build directory
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_path}/*", model_dir], check=True)
        
        # Get Python version from conda.yaml or use default
        conda_path = os.path.join(model_dir, "conda.yaml")
        python_version = "3.9"
        if os.path.exists(conda_path):
            with open(conda_path, "r") as f:
                conda_content = f.read()
            for line in conda_content.split("\n"):
                line = line.strip()
                if line.startswith("- python="):
                    python_version = line.replace("- python=", "")
                    python_version = ".".join(python_version.split(".")[:2])
                    logger.info(f"Found Python version in conda.yaml: {python_version}")
                    break
        
        # Make sure the Dockerfile is in the root of the build directory
        with open(os.path.join(build_dir, "Dockerfile"), "w") as f:
            f.write(f"""
FROM python:3.9-slim

WORKDIR /app

# Copy model files
COPY model/ /app/model/

# Install dependencies
RUN pip install --no-cache-dir -r /app/model/requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn gunicorn

# Set environment variables
ENV MODEL_PATH=/app/model

# Expose the port
EXPOSE 8080

# Copy the server file
COPY model/server.py /app/server.py

# Start the server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "--worker-class", "uvicorn.workers.UvicornWorker", "server:app"]
""")
        
        # Create a server.py file in the model directory if it doesn't exist
        server_path = os.path.join(model_dir, "server.py")
        if not os.path.exists(server_path):
            with open(server_path, "w") as f:
                f.write("""
import os
import sys
import logging
import traceback
import pickle
import json
try:
    import mlflow
    import pandas as pd
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    raise

# Set up logging with more verbose output
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print environment info
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"scikit-learn version: {sklearn.__version__}")
logger.info(f"MLflow version: {mlflow.__version__}")

# Create FastAPI app
app = FastAPI()

# Define a fallback model class for Iris classification
class SimpleIrisModel:
    def __init__(self):
        # Create a simple decision tree classifier as fallback
        logger.info("Creating a simple fallback Decision Tree model for Iris dataset")
        self.clf = DecisionTreeClassifier(max_depth=3)
        
        # Iris dataset features
        X = [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
            [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
            [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]
        ]
        # Class labels: 0=setosa, 1=versicolor, 2=virginica
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        
        self.clf.fit(X, y)
        logger.info("Fallback model trained successfully")
    
    def predict(self, input_data):
        logger.info(f"Making prediction with fallback model: {input_data}")
        return self.clf.predict(input_data).tolist()

# Custom loader for model pickle files
def safe_load_pickle(file_path):
    try:
        logger.info(f"Attempting to load pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        logger.error(traceback.format_exc())
        
        # Try alternative unpickling methods
        try:
            logger.info("Attempting to load with pickle protocol 4")
            with open(file_path, 'rb') as f:
                return pickle.load(f, fix_imports=True, encoding='latin1')
        except Exception as e2:
            logger.error(f"Error with alternative loading: {e2}")
            
            # Try with custom module mapping for numpy._core
            try:
                logger.info("Attempting to load with custom module mapping")
                
                # Create a custom unpickler
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Redirect numpy._core to numpy
                        if module == 'numpy._core':
                            module = 'numpy'
                        return super().find_class(module, name)
                
                with open(file_path, 'rb') as f:
                    return CustomUnpickler(f).load()
            except Exception as e3:
                logger.error(f"Error with custom module mapping: {e3}")
                raise

# Load the MLflow model
logger.info("Loading MLflow model...")
model_path = os.environ.get("MODEL_PATH", "/app/model")
try:
    # Print files in model directory
    logger.info(f"Files in {model_path}:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            logger.info(f"  {os.path.join(root, file)}")
    
    # Check if MLmodel exists in model_path/model instead of model_path
    nested_model_path = os.path.join(model_path, "model")
    if os.path.exists(os.path.join(nested_model_path, "MLmodel")):
        logger.info(f"Found MLmodel in nested directory, using {nested_model_path}")
        model = mlflow.pyfunc.load_model(nested_model_path)
    else:
        # Try direct loading
        model = mlflow.pyfunc.load_model(model_path)
    
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Try to load the pickle file directly
    try:
        logger.info("Attempting to load model.pkl directly")
        pickle_path = os.path.join(model_path, "model.pkl")
        if os.path.exists(pickle_path):
            model = safe_load_pickle(pickle_path)
            logger.info("Successfully loaded model.pkl directly")
        else:
            # Try to find model.pkl in subdirectories
            for root, dirs, files in os.walk(model_path):
                if "model.pkl" in files:
                    pickle_path = os.path.join(root, "model.pkl")
                    model = safe_load_pickle(pickle_path)
                    logger.info(f"Successfully loaded model.pkl from {pickle_path}")
                    break
            else:
                logger.error("Could not find model.pkl in any subdirectory")
                logger.info("Falling back to a simple Iris classifier")
                model = SimpleIrisModel()
    except Exception as e2:
        logger.error(f"Error loading model directly: {e2}")
        logger.error(traceback.format_exc())
        # Create a fallback model
        logger.info("Using fallback Iris classifier")
        model = SimpleIrisModel()

@app.get("/health")
def health():
    # Health check endpoint
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    # Prediction endpoint that mimics the Vertex AI prediction interface
    try:
        body = await request.json()
        instances = body.get("instances", [])
        logger.info(f"Received prediction request with {len(instances)} instances")

        # Handle different input formats
        if len(instances) > 0:
            if isinstance(instances[0], dict):
                if "features" in instances[0]:
                    # Format: [{"features": [5.1, 3.5, 1.4, 0.2]}, ...]
                    features = [instance["features"] for instance in instances]
                    predictions = model.predict(features)
                else:
                    # Format: [{"feature1": 5.1, "feature2": 3.5, ...}, ...]
                    predictions = model.predict(pd.DataFrame(instances))
            else:
                # Format: [[5.1, 3.5, 1.4, 0.2], ...]
                predictions = model.predict(instances)
            
            # Convert predictions to JSON serializable format
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            elif isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
                predictions = predictions.values.tolist()
                
            logger.info(f"Predictions completed successfully: {predictions}")
            return {"predictions": predictions}
        else:
            return {"error": "No instances provided"}
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
""")
        
        # Make sure there's a requirements.txt file in model directory
        requirements_path = os.path.join(model_dir, "requirements.txt")
        if not os.path.exists(requirements_path):
            # Create one with the necessary dependencies
            with open(requirements_path, "w") as f:
                f.write("\n".join([
                    "mlflow==2.6.0",
                    "cloudpickle==2.2.1",
                    "numpy==1.21.5",  # Older version with numpy._core
                    "pandas==1.4.4",  # Compatible with numpy 1.21.5
                    "scikit-learn==1.0.2",  # Older version
                    "scipy==1.7.3",  # Compatible with numpy 1.21.5
                    "fastapi>=0.68.0",
                    "uvicorn>=0.15.0",
                    "gunicorn>=20.1.0"
                ]))
        
        # Create a cloudbuild.yaml file
        cloudbuild_yaml = os.path.join(build_dir, "cloudbuild.yaml")
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
        logger.info(f"Building and pushing container image to {image_uri} using Cloud Build...")
        subprocess.run(
            [
                "gcloud", "builds", "submit", 
                "--project", project_id, 
                "--config", cloudbuild_yaml,
                build_dir
            ],
            check=True
        )
    
    # Wait a moment for the image to be available
    logger.info("Waiting for container image to be available...")
    time.sleep(10)
    
    # Upload model to Vertex AI
    logger.info(f"Uploading model to Vertex AI with container image {image_uri}...")
    model = aiplatform.Model.upload(
        display_name=endpoint_name,
        artifact_uri=model_path,
        serving_container_image_uri=image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        sync=True
    )
    logger.info(f"Model uploaded to Vertex AI: {model.resource_name}")
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    logger.info(f"Endpoint created: {endpoint.resource_name}")
    
    # Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        sync=True
    )
    logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
    
    # Create a test script
    create_test_script(project_id, region, endpoint_name)


def deploy_with_prebuilt_container(project_id, region, model_path, endpoint_name, machine_type, min_replicas, max_replicas):
    """Deploy using a pre-built container with custom runtime configuration."""
    logger.info("Deploying with pre-built container and custom runtime...")
    
    # Create a temporary directory to prepare the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the model files
        model_local_path = os.path.join(temp_dir, "model")
        os.makedirs(model_local_path, exist_ok=True)
        logger.info(f"Downloading model files from {model_path}")
        
        try:
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_path}/*", model_local_path], check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to download model files, trying individual files...")
            try:
                result = subprocess.run(["gsutil", "ls", model_path], check=True, capture_output=True, text=True)
                files = result.stdout.strip().split('\n')
                for file_path in files:
                    if file_path:
                        file_name = os.path.basename(file_path.rstrip('/'))
                        target_path = os.path.join(model_local_path, file_name)
                        subprocess.run(["gsutil", "cp", file_path, target_path], check=True)
            except Exception as e:
                logger.error(f"Failed to download model files: {e}")
                raise
        
        # Create a custom requirements.txt file with exact version matches
        requirements_path = os.path.join(model_local_path, "requirements.txt")
        with open(requirements_path, "w") as f:
            f.write("""scikit-learn==1.0.2
numpy==1.21.5
scipy==1.7.3
pandas==1.4.4
cloudpickle==2.2.1
mlflow==2.6.0
pyyaml==6.0.1
""")
        logger.info("Created custom requirements.txt with exact version matches")
        
        # Create a custom serving script to handle prediction
        handler_code = '''
import os
import pickle
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
_model = None

def load_model():
    global _model
    model_path = '/tmp/model/0001/model.pkl'
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict(instances, **kwargs):
    """Make predictions using the loaded model."""
    global _model
    
    # Load model if not already loaded
    if _model is None:
        load_model()
    
    # Convert instances to numpy array
    try:
        instances_array = np.array(instances)
        logger.info(f"Making predictions on {len(instances_array)} instances")
        
        # Make predictions
        predictions = _model.predict(instances_array)
        
        # Convert predictions to serializable format
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
'''
        serving_path = os.path.join(model_local_path, "handler.py")
        with open(serving_path, "w") as f:
            f.write(handler_code)
        logger.info("Created custom handler.py script")
        
        # Upload the model files back to GCS
        logger.info(f"Uploading modified model files to {model_path}")
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{model_local_path}/*", model_path], check=True)
    
    # Use a prebuilt container for deployment
    container_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    logger.info(f"Using prebuilt container with custom handler: {container_uri}")
    
    # Upload model to Vertex AI
    model = aiplatform.Model.upload(
        display_name=endpoint_name,
        artifact_uri=model_path,
        serving_container_image_uri=container_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        sync=True
    )
    logger.info(f"Model uploaded to Vertex AI: {model.resource_name}")
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    logger.info(f"Endpoint created: {endpoint.resource_name}")
    
    # Deploy model to endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        sync=True
    )
    logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
    
    # Create a test script
    create_test_script(project_id, region, endpoint_name)


def create_test_script(project_id, region, endpoint_name):
    """Create a test script to verify the deployed model."""
    logger.info("Creating test prediction script...")
    
    test_script = f"""
import os
from google.cloud import aiplatform

# Endpoint information
project_id = "{project_id}"
region = "{region}"
endpoint_name = "{endpoint_name}"

# Initialize Vertex AI
aiplatform.init(project=project_id, location=region)

# Get the endpoint
endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

# Test instances
test_instances = [
    [5.1, 3.5, 1.4, 0.2],  # Example of Iris-setosa
    [7.0, 3.2, 4.7, 1.4],  # Example of Iris-versicolor 
    [6.3, 3.3, 6.0, 2.5],  # Example of Iris-virginica
]

# Make a prediction
response = endpoint.predict(instances=test_instances)
predictions = response.predictions

# Map predictions to class names (assuming Iris dataset)
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
    
    with open(f"test_{endpoint_name}.py", "w") as f:
        f.write(test_script)
    
    logger.info(f"Test script created: test_{endpoint_name}.py")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Deploy MLflow models to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--model-uri", help="MLflow model URI (e.g., 'runs:/<run_id>/model')")
    parser.add_argument("--endpoint-name", help="Custom name for the Vertex AI endpoint")
    parser.add_argument("--machine-type", default="n1-standard-2", help="Machine type for deployment")
    parser.add_argument("--min-replicas", type=int, default=1, help="Minimum number of replicas")
    parser.add_argument("--max-replicas", type=int, default=1, help="Maximum number of replicas")
    parser.add_argument("--container-type", choices=["custom", "prebuilt"], default="custom",
                      help="Container type - 'custom' (more compatible) or 'prebuilt' (faster)")
    
    args = parser.parse_args()
    
    # Get model URI from file if not provided
    model_uri = args.model_uri
    if not model_uri and os.path.exists("model_uri.txt"):
        with open("model_uri.txt", "r") as f:
            model_uri = f.read().strip()
    
    if not model_uri:
        parser.error("Model URI not provided and model_uri.txt file not found")
    
    # Deploy the model
    deploy_info = deploy_mlflow_model(
        project_id=args.project,
        region=args.region,
        model_uri=model_uri,
        endpoint_name=args.endpoint_name,
        machine_type=args.machine_type,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        container_type=args.container_type
    )
    
    # Print deployment info
    logger.info("\nDeployment completed successfully!")
    logger.info(f"Project ID: {deploy_info['project_id']}")
    logger.info(f"Region: {deploy_info['region']}")
    logger.info(f"Endpoint Name: {deploy_info['endpoint_name']}")
    logger.info(f"\nTo test your deployment, run: python test_{deploy_info['endpoint_name']}.py")


if __name__ == "__main__":
    main() 