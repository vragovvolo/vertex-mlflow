# MLflow to Vertex AI Deployer

A tool for deploying MLflow models to Google Cloud Vertex AI with proper version compatibility handling.

## Problem

Deploying MLflow models to Vertex AI can be challenging due to version compatibility issues between:
- Python versions
- NumPy and SciPy versions
- scikit-learn versions
- Serialization formats

This tool solves these challenges by:
1. Creating appropriate compatibility layers
2. Building custom containers with the right versions
3. Providing fallback options for difficult cases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vertex_mlflow.git
cd vertex_mlflow

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure you have Google Cloud SDK installed and authenticated
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Usage

### Method 1: Deploy MLflow model directly

This method attempts to deploy your MLflow model directly to Vertex AI with appropriate version handling:

```bash
python mlflow_vertex_deployer.py \
  --project YOUR_GCP_PROJECT_ID \
  --region YOUR_GCP_REGION \
  --model-uri runs:/YOUR_MLFLOW_RUN_ID/model \
  --endpoint-name YOUR_ENDPOINT_NAME \
  --container-type custom
```

Parameters:
- `--project`: Your Google Cloud Project ID
- `--region`: Google Cloud region (e.g., us-central1)
- `--model-uri`: MLflow model URI (e.g., runs:/<run_id>/model)
- `--endpoint-name`: Name for the endpoint (optional)
- `--container-type`: "custom" (default) or "prebuilt"
- `--machine-type`: Machine type (default: n1-standard-2)
- `--min-replicas`: Minimum replicas (default: 1)
- `--max-replicas`: Maximum replicas (default: 1)

### Method 2: Deploy a simple model (for compatibility issues)

If you encounter persistent compatibility issues, this alternative approach deploys a simplified version of your model:

```bash
python examples/iris/deploy_simple_iris.py \
  --project YOUR_GCP_PROJECT_ID \
  --region YOUR_GCP_REGION \
  --endpoint-name YOUR_ENDPOINT_NAME
```

## Testing your deployment

After deploying, a test script is automatically generated for you. Run it to verify your deployment:

```bash
# Test the main deployment
python test_YOUR_ENDPOINT_NAME.py

# For comprehensive testing of Iris models
python examples/iris/test_iris_comprehensive.py
```

## Troubleshooting

### Common issues

- **numpy._core module error**: This occurs when the numpy version in the deployment container is incompatible with your model. Solutions:
  - Try using Method 2 (simple model approach)
  - Specify compatible numpy version in your MLmodel requirements
  
- **Model loading errors**: Check Vertex AI logs for detailed error messages and use the appropriate compatibility options.

- **Serialization errors**: These can happen with older models. Try the custom unpickler options in the deployment script.

### Viewing logs

```bash
# View logs for your endpoint
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id=YOUR_ENDPOINT_ID"
```

## Examples

### Iris Classification Model

Check the `examples/iris/` directory for a complete example of:
- A simple Iris classifier
- Deployment scripts
- Comprehensive testing

## How it works

1. **Model Extraction**: The tool downloads your MLflow model and its dependencies
2. **Compatibility Analysis**: It analyzes your model's requirements and creates appropriate compatibility files
3. **Container Building**: It builds a custom container with the right dependency versions
4. **Deployment**: It deploys the container to Vertex AI and creates an endpoint
5. **Verification**: It generates a test script for you to verify the deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.
