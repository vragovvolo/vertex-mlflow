import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from vertex_mlflow.utils import generate_dockerfile, build_docker_image, MockCloudBuildClient


class TestUtils(unittest.TestCase):

    def test_generate_dockerfile(self):
        """Test that generate_dockerfile creates a file with the correct content."""
        model_uri = "models:/my-model/1"
        dockerfile_path = generate_dockerfile(model_uri)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(dockerfile_path))
        
        # Check that the file has the expected content
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            self.assertIn("FROM python:3.9-slim", content)
            self.assertIn("pip install --no-cache-dir mlflow==2.7.0", content)
            self.assertIn('EXPOSE 8080', content)
        
        # Clean up
        os.remove(dockerfile_path)
        os.rmdir(os.path.dirname(dockerfile_path))

    @patch('vertex_mlflow.utils.MockCloudBuildClient')
    def test_build_docker_image(self, mock_client_class):
        """Test that build_docker_image calls Cloud Build correctly."""
        # Setup mock
        mock_client = MagicMock()
        mock_build_operation = MagicMock()
        mock_build_operation.result.return_value = None
        mock_client.create_build.return_value = mock_build_operation
        mock_client_class.return_value = mock_client
        
        # Call the function
        with tempfile.NamedTemporaryFile() as tmp:
            result = build_docker_image(
                project="test-project",
                region="us-central1",
                dockerfile_path=tmp.name,
                artifact_registry_repo="us-central1-docker.pkg.dev/test-project/test-repo",
                image_tag="test-image"
            )
        
        # Assertions
        self.assertEqual(result, "us-central1-docker.pkg.dev/test-project/test-repo/test-image:latest")
        mock_client.create_build.assert_called_once()
        mock_build_operation.result.assert_called_once()


if __name__ == '__main__':
    unittest.main() 