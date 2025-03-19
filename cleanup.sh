#!/bin/bash
# Cleanup script for MLflow-Vertex AI Deployer

# Core files to keep
CORE_FILES=(
    "mlflow_vertex_deployer.py"
    "README.md"
    "requirements.txt"
    "setup.py"
    "pyproject.toml"
    "tests"
    "examples"
    "mlruns"
    "venv"
    "cleanup.sh"
)

# Make sure we're in the right directory
if [[ ! -f "mlflow_vertex_deployer.py" ]]; then
    echo "Error: Run this script from the root of the MLflow-Vertex AI Deployer directory"
    exit 1
fi

echo "Cleaning up unnecessary files..."

# Find all files and directories in the current directory
for file in *; do
    # Skip if it's a core file to keep
    if [[ " ${CORE_FILES[@]} " =~ " ${file} " ]]; then
        echo "Keeping: $file"
        continue
    fi
    
    # Handle directories specially - don't delete them without checking
    if [[ -d "$file" ]]; then
        if [[ "$file" == "google-cloud-sdk" || "$file" == "google-cloud-cli"* ]]; then
            echo "Skipping Google Cloud SDK directory: $file"
            continue
        fi
        
        if [[ "$file" == "vertex_mlflow" || "$file" == "vertex_mlflow.egg-info" ]]; then
            echo "Keeping package directory: $file"
            continue
        fi
        
        if [[ "$file" == "compatible_model" ]]; then
            echo "Keeping model directory: $file"
            continue
        fi
    fi
    
    # Check if it's a deployment script not needed anymore
    if [[ "$file" == "deploy_"* && "$file" != "mlflow_vertex_deployer.py" ]]; then
        echo "Removing deprecated deployment script: $file"
        rm -f "$file"
        continue
    fi
    
    # Check if it's an old test or prediction script
    if [[ "$file" == "test_"* || "$file" == "predict_"* || "$file" == "make_prediction.py" || "$file" == "manual_test_prediction.py" ]]; then
        echo "Removing old test/prediction script: $file"
        rm -f "$file"
        continue
    fi
    
    # Check if it's an old model creation script
    if [[ "$file" == "create_"* ]]; then
        echo "Removing old model creation script: $file"
        rm -f "$file"
        continue
    fi
    
    # Don't delete remaining directories automatically
    if [[ -d "$file" ]]; then
        echo "Keeping other directory: $file"
        continue
    fi
    
    # Ask for confirmation for any other files
    read -p "Remove $file? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing: $file"
        rm -f "$file"
    else
        echo "Keeping: $file"
    fi
done

echo "Cleanup complete!"

# Create examples directory if it doesn't exist
if [[ ! -d "examples" ]]; then
    echo "Creating examples directory..."
    mkdir -p examples
fi

# Create a simple example script in the examples directory
if [[ ! -f "examples/deploy_iris_model.py" ]]; then
    echo "Creating example script for Iris model deployment..."
    cat > examples/deploy_iris_model.py << 'EOF'
"""
Example script to train and deploy an Iris classification model to Vertex AI.
"""
import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_vertex_deployer import deploy_mlflow_model

def train_and_log_model():
    """Train a simple model on the Iris dataset and log it to MLflow."""
    # Set up MLflow
    mlflow.set_tracking_uri("file:" + os.path.abspath("../mlruns"))
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metrics({"train_score": train_score, "test_score": test_score})
        
        return run.info.run_id

def main():
    # Train model and get run ID
    run_id = train_and_log_model()
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Model trained and logged with URI: {model_uri}")
    
    # Save model URI to file
    with open("model_uri.txt", "w") as f:
        f.write(model_uri)
    
    # Deploy to Vertex AI (uncomment and add your project ID to deploy)
    # deploy_info = deploy_mlflow_model(
    #     project_id="your-gcp-project-id",
    #     region="us-central1",
    #     model_uri=model_uri,
    #     endpoint_name="iris-example"
    # )
    # 
    # print(f"Model deployed to Vertex AI endpoint: {deploy_info['endpoint_name']}")

if __name__ == "__main__":
    main()
EOF
    echo "Example script created at examples/deploy_iris_model.py"
fi

echo "Setup complete! You can now use mlflow_vertex_deployer.py to deploy MLflow models to Vertex AI." 