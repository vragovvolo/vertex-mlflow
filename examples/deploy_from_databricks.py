"""
Example script to deploy a model from Databricks to Vertex AI.
"""
import os
import sys
import argparse

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_vertex_deployer import deploy_mlflow_model


def deploy_databricks_model(project_id, region, model_uri, endpoint_name=None):
    """
    Deploy a model from Databricks to Vertex AI.
    
    Args:
        project_id (str): Google Cloud project ID
        region (str): Google Cloud region
        model_uri (str): Databricks MLflow model URI (e.g., 'models:/MyModel/1')
        endpoint_name (str, optional): Name for the Vertex AI endpoint
    """
    # Ensure Databricks credentials are available
    if "DATABRICKS_HOST" not in os.environ or "DATABRICKS_TOKEN" not in os.environ:
        raise ValueError(
            "Databricks credentials not found. Please set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables."
        )
    
    print(f"Deploying Databricks model {model_uri} to Vertex AI in {project_id}...")
    
    # Deploy the model to Vertex AI
    deploy_info = deploy_mlflow_model(
        project_id=project_id,
        region=region,
        model_uri=model_uri,
        endpoint_name=endpoint_name,
        container_type="custom"  # Custom container for maximum compatibility
    )
    
    print("\nDeployment completed successfully!")
    print(f"Project ID: {deploy_info['project_id']}")
    print(f"Region: {deploy_info['region']}")
    print(f"Endpoint Name: {deploy_info['endpoint_name']}")
    print(f"\nTo test your deployment, run: python test_{deploy_info['endpoint_name']}.py")
    
    return deploy_info


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Deploy Databricks models to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--model-uri", required=True, 
                      help="Databricks MLflow model URI (e.g., 'models:/MyModel/1')")
    parser.add_argument("--endpoint-name", help="Custom name for the Vertex AI endpoint")
    
    args = parser.parse_args()
    
    # Deploy the model
    deploy_databricks_model(
        project_id=args.project,
        region=args.region,
        model_uri=args.model_uri,
        endpoint_name=args.endpoint_name
    )


if __name__ == "__main__":
    main() 