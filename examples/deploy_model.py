"""
A sample script demonstrating how to use the vertex-mlflow package
to deploy an MLflow model to Vertex AI.
"""

import os
import argparse
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from vertex_mlflow import deploy_model, predict


def train_and_log_model():
    """Train a simple model on the Iris dataset and log it to MLflow."""
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Log the model to MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
    
    print(f"Model logged to MLflow with run_id: {run_id}")
    return f"runs:/{run_id}/model"


def main():
    parser = argparse.ArgumentParser(description="Deploy an MLflow model to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="GCP region (e.g., us-central1)")
    parser.add_argument("--model-uri", help="MLflow model URI (e.g., 'runs:/<run_id>/model')")
    parser.add_argument("--artifact-registry-repo", required=True, help="Artifact Registry repo URI")
    parser.add_argument("--name", default="mlflow-model", help="Name for the deployment")
    args = parser.parse_args()
    
    # Train and log a model if model_uri is not provided
    model_uri = args.model_uri or train_and_log_model()
    
    print(f"Deploying model from {model_uri} to Vertex AI...")
    
    # Deploy the model
    deployed_model = deploy_model(
        name=args.name,
        model_uri=model_uri,
        project=args.project,
        region=args.region,
        artifact_registry_repo=args.artifact_registry_repo
    )
    
    print(f"Model deployed successfully to endpoint: {deployed_model.resource_name}")
    
    # Make a prediction
    iris = load_iris()
    instances = [{"features": iris.data[0].tolist()}]
    
    print("Making a prediction with the deployed model...")
    prediction = predict(
        project=args.project,
        region=args.region,
        endpoint_name=args.name,  # Assuming endpoint has the same name
        instances=instances
    )
    
    print(f"Prediction result: {prediction}")


if __name__ == "__main__":
    main() 