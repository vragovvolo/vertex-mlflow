import unittest
from unittest.mock import patch, MagicMock
from vertex_mlflow.registry import register_model


class TestRegistry(unittest.TestCase):
    
    @patch('vertex_mlflow.registry.aiplatform.init')
    @patch('vertex_mlflow.registry.aiplatform.Model.upload')
    def test_register_model(self, mock_model_upload, mock_init):
        """Test registering a model."""
        # Setup mocks
        mock_init.return_value = None
        mock_model = MagicMock()
        mock_model.resource_name = "projects/test-project/locations/us-central1/models/123"
        mock_model_upload.return_value = mock_model
        
        # Call the function
        result = register_model(
            model_uri="models:/test-model/1",
            name="test-model",
            project="test-project",
            region="us-central1",
            destination_image_uri="us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest",
            labels={"key": "value"}
        )
        
        # Assert init was called
        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        
        # Assert model upload was called with the right arguments
        mock_model_upload.assert_called_once_with(
            display_name="test-model",
            artifact_uri="models:/test-model/1",
            serving_container_image_uri="us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest",
            labels={"key": "value"},
            sync=True
        )
        
        # Assert the result is the model from the upload
        self.assertEqual(result, mock_model)
        
    @patch('vertex_mlflow.registry.aiplatform.init')
    @patch('vertex_mlflow.registry.aiplatform.Model.upload')
    def test_register_model_no_labels(self, mock_model_upload, mock_init):
        """Test registering a model without labels."""
        # Setup mocks
        mock_init.return_value = None
        mock_model = MagicMock()
        mock_model_upload.return_value = mock_model
        
        # Call the function
        result = register_model(
            model_uri="models:/test-model/1",
            name="test-model",
            project="test-project",
            region="us-central1",
            destination_image_uri="us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest"
        )
        
        # Assert model upload was called with an empty labels dict
        mock_model_upload.assert_called_once_with(
            display_name="test-model",
            artifact_uri="models:/test-model/1",
            serving_container_image_uri="us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest",
            labels={},
            sync=True
        )


if __name__ == '__main__':
    unittest.main() 