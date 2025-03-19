import os
import json
import time
import logging
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPIError

from .utils import build_docker_image, generate_dockerfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def deploy_model(
    name,
    model_uri,
    project,
    region,
    machine_type="n1-standard-4",
    accelerator_type=None,
    accelerator_count=0,
    destination_image_uri=None,
    artifact_registry_repo=None,
    enable_monitoring=False,
    vpc_network=None,
    service_account=None,
    endpoint_name=None,
    traffic_percentage=100,
    extra_deploy_params=None
):
    """
    Deploys an MLflow model to Vertex AI.
    
    Parameters:
      - name: Deployment name (used for endpoint display name if new).
      - model_uri: MLflow model URI (e.g. "models:/MyModel/1" or "runs:/...").
      - project: GCP project ID.
      - region: GCP region (e.g., "us-central1").
      - machine_type: Compute instance type.
      - accelerator_type: Optional accelerator type (e.g. "NVIDIA_TESLA_T4").
      - accelerator_count: Number of accelerators.
      - destination_image_uri: Fully qualified image URI in Artifact Registry.
      - artifact_registry_repo: Artifact Registry repository URI if building new image.
      - enable_monitoring: If True, enable Vertex AI Model Monitoring.
      - vpc_network: Full VPC network resource name (for private endpoints).
      - service_account: Service account email to use for deployment.
      - endpoint_name: If provided, deploy to an existing endpoint.
      - traffic_percentage: Percent of traffic for this model (if updating an endpoint).
      - extra_deploy_params: Dictionary for additional deployment parameters.
      
    Returns:
      The deployed Vertex AI Model resource.
    """

    # Initialize the Vertex AI SDK.
    aiplatform.init(project=project, location=region)

    # Step 1: Build the Docker image if not provided.
    if not destination_image_uri:
        if not artifact_registry_repo:
            raise ValueError("Either destination_image_uri or artifact_registry_repo must be provided.")
        logger.info("Generating Dockerfile from MLflow model: %s", model_uri)
        dockerfile_path = generate_dockerfile(model_uri)
        # Build docker image via Cloud Build.
        destination_image_uri = build_docker_image(
            project=project,
            region=region,
            dockerfile_path=dockerfile_path,
            artifact_registry_repo=artifact_registry_repo,
            image_tag=name
        )
        logger.info("Built Docker image: %s", destination_image_uri)

    # Step 2: Upload the model artifact to Vertex AI Model Registry.
    logger.info("Uploading model artifact to Vertex AI Model Registry...")
    try:
        model = aiplatform.Model.upload(
            display_name=name,
            artifact_uri=model_uri,
            serving_container_image_uri=destination_image_uri,
            sync=True,
            labels={"mlflow_model_uri": model_uri}
        )
    except GoogleAPIError as e:
        logger.error("Error uploading model: %s", e)
        raise

    # Step 3: Get or create the endpoint.
    endpoint = None
    if endpoint_name:
        logger.info("Retrieving existing endpoint: %s", endpoint_name)
        endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
        if endpoints:
            endpoint = endpoints[0]
        else:
            logger.warning("Endpoint %s not found. A new endpoint will be created.", endpoint_name)
    if endpoint is None:
        logger.info("Creating new endpoint: %s", name)
        try:
            endpoint = aiplatform.Endpoint.create(
                display_name=name,
                network=vpc_network,  # May be None.
                sync=True
            )
        except GoogleAPIError as e:
            logger.error("Error creating endpoint: %s", e)
            raise

    # Step 4: Deploy the model to the endpoint.
    logger.info("Deploying model to endpoint...")
    try:
        deployed_model = model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=1,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            service_account=service_account,
            traffic_split={model.resource_name: traffic_percentage},
            sync=True,
            **(extra_deploy_params or {})
        )
    except GoogleAPIError as e:
        logger.error("Error deploying model: %s", e)
        raise

    # Optionally enable model monitoring.
    if enable_monitoring:
        try:
            from .monitoring import enable_model_monitoring
            enable_model_monitoring(endpoint, deployed_model)
        except Exception as e:
            logger.warning("Failed to enable model monitoring: %s", e)

    logger.info("Deployment successful. Endpoint: %s", endpoint.resource_name)
    return deployed_model

def update_deployment(
    name,
    model_uri,
    project,
    region,
    endpoint_name,
    **update_params
):
    """
    Updates an existing Vertex AI deployment with a new model version.
    
    This function redeploys a model to an existing endpoint.
    """
    aiplatform.init(project=project, location=region)
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if not endpoints:
        raise ValueError(f"Endpoint {endpoint_name} not found.")
    endpoint = endpoints[0]
    
    # Here we deploy a new model version to the same endpoint.
    logger.info("Updating deployment for endpoint: %s", endpoint_name)
    new_deployed_model = deploy_model(
        name=name,
        model_uri=model_uri,
        project=project,
        region=region,
        endpoint_name=endpoint_name,
        **update_params
    )
    return new_deployed_model

def delete_deployment(project, region, endpoint_name, deployed_model_id=None):
    """
    Deletes a deployment from Vertex AI.
    
    If deployed_model_id is provided, only undeploy that model. Otherwise,
    deletes the entire endpoint.
    """
    aiplatform.init(project=project, location=region)
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if not endpoints:
        raise ValueError(f"Endpoint {endpoint_name} not found.")
    endpoint = endpoints[0]
    if deployed_model_id:
        logger.info("Undeploying model %s from endpoint %s", deployed_model_id, endpoint_name)
        endpoint.undeploy(deployed_model_id=deployed_model_id, sync=True)
    else:
        logger.info("Deleting entire endpoint: %s", endpoint_name)
        endpoint.delete()
    logger.info("Deletion complete.")

def list_deployments(project, region, filter_str=None):
    """
    Lists Vertex AI endpoints (deployments) for the given project and region.
    Optionally filter by display name.
    """
    aiplatform.init(project=project, location=region)
    endpoints = aiplatform.Endpoint.list(filter=filter_str)
    deployments = []
    for ep in endpoints:
        deployments.append({
            "display_name": ep.display_name,
            "resource_name": ep.resource_name,
            "deployed_models": ep.deployed_models,
        })
    return deployments

def predict(project, region, endpoint_name, instances):
    """
    Sends a prediction request to the specified Vertex AI endpoint.
    
    Parameters:
      - project: GCP project ID.
      - region: GCP region.
      - endpoint_name: Display name of the endpoint.
      - instances: A list of input instances (dictionaries).
      
    Returns:
      The prediction results.
    """
    aiplatform.init(project=project, location=region)
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if not endpoints:
        raise ValueError(f"Endpoint {endpoint_name} not found.")
    endpoint = endpoints[0]
    response = endpoint.predict(instances=instances)
    return response
