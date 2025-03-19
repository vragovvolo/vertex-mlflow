import logging
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)

def register_model(model_uri, name, project, region, destination_image_uri, labels=None):
    """
    Registers an MLflow model with Vertex AI Model Registry.
    
    Parameters:
      - model_uri: MLflow model URI.
      - name: Display name for the model.
      - project: GCP project ID.
      - region: GCP region.
      - destination_image_uri: Container image URI to serve the model.
      - labels: Optional dictionary of labels.
      
    Returns:
      The Vertex AI Model resource.
    """
    aiplatform.init(project=project, location=region)
    try:
        model = aiplatform.Model.upload(
            display_name=name,
            artifact_uri=model_uri,
            serving_container_image_uri=destination_image_uri,
            labels=labels or {},
            sync=True
        )
        logger.info("Model registered successfully: %s", model.resource_name)
        return model
    except GoogleAPIError as e:
        logger.error("Error registering model: %s", e)
        raise
