"""
Unit tests for the MLflow Vertex AI Deployer.
"""
import os
import unittest
from unittest import mock
import tempfile
import shutil

# Import the module to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow_vertex_deployer


class TestMLflowVertexDeployer(unittest.TestCase):
    """Tests for MLflow Vertex AI Deployer."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @mock.patch('mlflow_vertex_deployer.mlflow')
    @mock.patch('mlflow_vertex_deployer.aiplatform')
    @mock.patch('mlflow_vertex_deployer.storage')
    @mock.patch('mlflow_vertex_deployer.subprocess.run')
    def test_deploy_mlflow_model_prebuilt(self, mock_subprocess, mock_storage, mock_aiplatform, mock_mlflow):
        """Test deploying a model with a prebuilt container."""
        # Mock storage bucket
        mock_bucket = mock.MagicMock()
        mock_storage_client = mock.MagicMock()
        mock_storage_client.get_bucket.return_value = mock_bucket
        mock_storage.Client.return_value = mock_storage_client
        
        # Mock Vertex AI components
        mock_model = mock.MagicMock()
        mock_endpoint = mock.MagicMock()
        mock_aiplatform.Model.upload.return_value = mock_model
        mock_aiplatform.Endpoint.create.return_value = mock_endpoint
        
        # Call the function
        result = mlflow_vertex_deployer.deploy_mlflow_model(
            project_id="test-project",
            region="us-central1",
            model_uri="runs:/12345abcdef/model",
            endpoint_name="test-endpoint",
            container_type="prebuilt"
        )
        
        # Assertions
        self.assertEqual(result["project_id"], "test-project")
        self.assertEqual(result["region"], "us-central1")
        self.assertEqual(result["endpoint_name"], "test-endpoint")
        
        # Verify Vertex AI was initialized
        mock_aiplatform.init.assert_called_once_with(project="test-project", location="us-central1")
        
        # Verify model was downloaded
        mock_mlflow.artifacts.download_artifacts.assert_called_once()
        
        # Verify model was uploaded to Vertex AI
        mock_aiplatform.Model.upload.assert_called_once()
        
        # Verify endpoint was created
        mock_aiplatform.Endpoint.create.assert_called_once()
        
        # Verify model was deployed
        mock_model.deploy.assert_called_once()

    @mock.patch('mlflow_vertex_deployer.create_compatibility_files')
    @mock.patch('mlflow_vertex_deployer.create_server_files')
    @mock.patch('mlflow_vertex_deployer.deploy_with_custom_container')
    @mock.patch('mlflow_vertex_deployer.mlflow')
    @mock.patch('mlflow_vertex_deployer.aiplatform')
    @mock.patch('mlflow_vertex_deployer.storage')
    @mock.patch('mlflow_vertex_deployer.subprocess.run')
    def test_deploy_mlflow_model_custom(self, mock_subprocess, mock_storage, mock_aiplatform, 
                                     mock_mlflow, mock_deploy_custom, mock_create_server, 
                                     mock_create_compat):
        """Test deploying a model with a custom container."""
        # Mock storage bucket
        mock_bucket = mock.MagicMock()
        mock_storage_client = mock.MagicMock()
        mock_storage_client.get_bucket.return_value = mock_bucket
        mock_storage.Client.return_value = mock_storage_client
        
        # Call the function
        result = mlflow_vertex_deployer.deploy_mlflow_model(
            project_id="test-project",
            region="us-central1",
            model_uri="runs:/12345abcdef/model",
            endpoint_name="test-endpoint",
            container_type="custom"
        )
        
        # Assertions
        self.assertEqual(result["project_id"], "test-project")
        self.assertEqual(result["region"], "us-central1")
        self.assertEqual(result["endpoint_name"], "test-endpoint")
        
        # Verify compatibility files were created
        mock_create_compat.assert_called_once()
        
        # Verify server files were created
        mock_create_server.assert_called_once()
        
        # Verify custom container deployment was called
        mock_deploy_custom.assert_called_once()


if __name__ == '__main__':
    unittest.main() 