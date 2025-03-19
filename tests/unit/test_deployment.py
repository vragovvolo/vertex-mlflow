import unittest
from unittest.mock import patch, MagicMock
from vertex_mlflow.deployment import deploy_model, list_deployments, predict


class TestDeployment(unittest.TestCase):
    
    @patch('vertex_mlflow.deployment.aiplatform.init')
    @patch('vertex_mlflow.deployment.aiplatform.Model.upload')
    @patch('vertex_mlflow.deployment.aiplatform.Endpoint.create')
    def test_deploy_model_new_endpoint(self, mock_endpoint_create, mock_model_upload, mock_init):
        """Test deploying a model to a new endpoint."""
        # Setup mocks
        mock_init.return_value = None
        mock_model = MagicMock()
        mock_model_upload.return_value = mock_model
        mock_endpoint = MagicMock()
        mock_endpoint_create.return_value = mock_endpoint
        mock_deployed_model = MagicMock()
        mock_model.deploy.return_value = mock_deployed_model
        
        # Call the function
        result = deploy_model(
            name="test-model",
            model_uri="models:/test-model/1",
            project="test-project",
            region="us-central1",
            destination_image_uri="us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest",
        )
        
        # Assert init was called
        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        
        # Assert model upload was called
        mock_model_upload.assert_called_once()
        
        # Assert endpoint create was called
        mock_endpoint_create.assert_called_once()
        
        # Assert model deploy was called
        mock_model.deploy.assert_called_once()
        
        # Assert result is the deployed model
        self.assertEqual(result, mock_deployed_model)
    
    @patch('vertex_mlflow.deployment.aiplatform.init')
    @patch('vertex_mlflow.deployment.aiplatform.Endpoint.list')
    def test_list_deployments(self, mock_endpoint_list, mock_init):
        """Test listing deployments."""
        # Setup mocks
        mock_init.return_value = None
        mock_endpoint1 = MagicMock()
        mock_endpoint1.display_name = "endpoint1"
        mock_endpoint1.resource_name = "projects/test/locations/us-central1/endpoints/1"
        mock_endpoint1.deployed_models = [{"id": "model1", "display_name": "Model 1"}]
        
        mock_endpoint2 = MagicMock()
        mock_endpoint2.display_name = "endpoint2"
        mock_endpoint2.resource_name = "projects/test/locations/us-central1/endpoints/2"
        mock_endpoint2.deployed_models = [{"id": "model2", "display_name": "Model 2"}]
        
        mock_endpoint_list.return_value = [mock_endpoint1, mock_endpoint2]
        
        # Call the function
        result = list_deployments(project="test-project", region="us-central1")
        
        # Assert init was called
        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        
        # Assert endpoint list was called
        mock_endpoint_list.assert_called_once_with(filter=None)
        
        # Assert the result contains the expected endpoints
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["display_name"], "endpoint1")
        self.assertEqual(result[1]["display_name"], "endpoint2")
    
    @patch('vertex_mlflow.deployment.aiplatform.init')
    @patch('vertex_mlflow.deployment.aiplatform.Endpoint.list')
    def test_predict(self, mock_endpoint_list, mock_init):
        """Test making predictions."""
        # Setup mocks
        mock_init.return_value = None
        mock_endpoint = MagicMock()
        mock_endpoint.predict.return_value = {"predictions": [1, 2, 3]}
        mock_endpoint_list.return_value = [mock_endpoint]
        
        # Call the function
        result = predict(
            project="test-project",
            region="us-central1",
            endpoint_name="test-endpoint",
            instances=[{"feature1": 1, "feature2": 2}]
        )
        
        # Assert init was called
        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        
        # Assert endpoint list was called with the right filter
        mock_endpoint_list.assert_called_once_with(filter="display_name=test-endpoint")
        
        # Assert predict was called with the right instances
        mock_endpoint.predict.assert_called_once_with(instances=[{"feature1": 1, "feature2": 2}])
        
        # Assert the result is the prediction from the endpoint
        self.assertEqual(result, {"predictions": [1, 2, 3]})


if __name__ == '__main__':
    unittest.main() 