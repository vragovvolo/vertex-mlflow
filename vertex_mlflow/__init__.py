"""
vertex_mlflow package – Integrates MLflow with Google Cloud Vertex AI.
Provides both a Python API and a CLI entry point.
"""

from .deployment import deploy_model, update_deployment, delete_deployment, list_deployments, predict
from .registry import register_model
from .monitoring import enable_model_monitoring

__version__ = "0.1.0"