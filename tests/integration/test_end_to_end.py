import os
import unittest
import tempfile
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch, MagicMock

from vertex_mlflow import deploy_model, predict


class TestEndToEnd(unittest.TestCase):
    """Integration tests for vertex-mlflow.
    
    These tests require:
    1. A GCP project with Vertex AI enabled
    2. Proper authentication (service account JSON or gcloud auth)
    3. MLflow tracking server (could be local or remote)
    
    To run these tests, set the following environment variables:
    - GCP_PROJECT: Your Google Cloud project ID
    - GCP_REGION: The region to deploy to (e.g., us-central1)
    - ARTIFACT_REGISTRY_REPO: Your Artifact Registry repo URI
    """
    
    @classmethod
    def setUpClass(cls):
        """Create a sample MLflow model for testing."""
        # Check if we have the required environment variables
        cls.project = os.environ.get('GCP_PROJECT')
        cls.region = os.environ.get('GCP_REGION')
        cls.artifact_repo = os.environ.get('ARTIFACT_REGISTRY_REPO')
        
        # Skip actual deployment if not running in an environment with GCP credentials
        if not all([cls.project, cls.region, cls.artifact_repo]):
            print("Skipping actual deployment tests - environment variables not set")
            cls.skip_deployment = True
            return
        else:
            cls.skip_deployment = False
        
        # Create a temporary directory for MLflow artifacts
        cls.temp_dir = tempfile.TemporaryDirectory()
        mlflow.set_tracking_uri(f"file://{cls.temp_dir.name}")
        
        # Train and log a simple model
        with mlflow.start_run() as run:
            # Load iris dataset
            iris = load_iris()
            X = iris.data
            y = iris.target
            
            # Train a model
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X, y)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            # Log the signature
            input_example = pd.DataFrame(iris.data[:3], columns=iris.feature_names)
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(model, "model-with-signature", signature=signature, input_example=input_example)
            
            # Save the run ID
            cls.run_id = run.info.run_id
            
            # Set model URI
            cls.model_uri = f"runs:/{cls.run_id}/model-with-signature"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    @unittest.skipIf(os.environ.get('SKIP_INTEGRATION_TESTS', 'true').lower() == 'true',
                     "Integration tests are disabled")
    def test_deploy_and_predict(self):
        """Test the end-to-end deployment and prediction flow.
        
        This test is marked to be skipped by default. To run it, set:
        SKIP_INTEGRATION_TESTS=false
        """
        if self.skip_deployment:
            self.skipTest("Required environment variables not set")
        
        # Use a unique model name for this test
        import uuid
        model_name = f"test-model-{uuid.uuid4().hex[:8]}"
        
        try:
            # Deploy the model
            deployed_model = deploy_model(
                name=model_name,
                model_uri=self.model_uri,
                project=self.project,
                region=self.region,
                artifact_registry_repo=self.artifact_repo,
                machine_type="n1-standard-4"
            )
            
            # Get the endpoint name from the deployed model
            endpoint_name = model_name  # We're using the same name for both
            
            # Verify endpoint exists
            self.assertIsNotNone(deployed_model)
            
            # Wait for deployment to be ready (in a real test, we'd wait for the endpoint to be ready)
            # Here we'll just assume it's ready immediately
            
            # Make a prediction
            iris = load_iris()
            instances = [{"features": iris.data[0].tolist()}]
            
            prediction = predict(
                project=self.project,
                region=self.region,
                endpoint_name=endpoint_name,
                instances=instances
            )
            
            # Verify we got a prediction
            self.assertIsNotNone(prediction)
            
        finally:
            # Clean up - in a real test, we'd want to delete the endpoint
            # from vertex_mlflow import delete_deployment
            # delete_deployment(project=self.project, region=self.region, endpoint_name=model_name)
            pass
    
    def test_mock_deploy_and_predict(self):
        """Test the deployment and prediction flow with mocks."""
        # This test uses mocks instead of real GCP resources
        with patch('vertex_mlflow.deployment.aiplatform.init'), \
             patch('vertex_mlflow.deployment.aiplatform.Model.upload') as mock_upload, \
             patch('vertex_mlflow.deployment.aiplatform.Endpoint.create') as mock_endpoint_create, \
             patch('vertex_mlflow.deployment.build_docker_image') as mock_build_image:
            
            # Setup mocks
            mock_model = MagicMock()
            mock_upload.return_value = mock_model
            mock_endpoint = MagicMock()
            mock_endpoint_create.return_value = mock_endpoint
            mock_deployed_model = MagicMock()
            mock_model.deploy.return_value = mock_deployed_model
            mock_build_image.return_value = "gcr.io/test-project/test-image:latest"
            
            # Deploy the model
            deploy_model(
                name="test-model",
                model_uri=self.model_uri,
                project="test-project",
                region="us-central1",
                artifact_registry_repo="us-central1-docker.pkg.dev/test-project/test-repo"
            )
            
            # Assert that the model was uploaded
            mock_upload.assert_called_once()
            
            # Assert that the endpoint was created
            mock_endpoint_create.assert_called_once()
            
            # Assert that the model was deployed
            mock_model.deploy.assert_called_once()


if __name__ == '__main__':
    unittest.main() 