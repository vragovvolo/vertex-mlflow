# Vertex-MLflow

**Vertex-MLflow** is a Python library that integrates MLflow with Google Cloud Vertex AI.
It provides both a CLI and Python API to deploy, update, delete, list, and predict using MLflow models on Vertex AI,
leveraging features like the Vertex AI Model Registry, Model Monitoring, multi-model endpoints, Artifact Registry, and secure deployments.

## Features

- **Model Registry Integration:** Register and version your MLflow models in Vertex AI.
- **Model Deployment:** Deploy MLflow models to Vertex AI endpoints with one command.
- **Model Monitoring:** Optionally enable Vertex AI Model Monitoring to track drift and skew.
- **Multi-Model Endpoints:** Support for deploying multiple models to a single endpoint with traffic splitting.
- **Secure Deployments:** Supports private VPC endpoints and IAM-based role controls.
- **CLI & Python API:** Use the provided CLI or import the library in your Python code.
