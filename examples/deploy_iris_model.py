"""
Example script to train and deploy an Iris classification model to Vertex AI.
"""
import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_vertex_deployer import deploy_mlflow_model

def train_and_log_model():
    """Train a simple model on the Iris dataset and log it to MLflow."""
    # Set up MLflow
    mlflow_tracking_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    print(f"Using MLflow tracking at: {mlflow_tracking_dir}")
    
    # Create or get experiment
    experiment_name = "iris_classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metrics({"train_score": train_score, "test_score": test_score})
        
        return run.info.run_id

def main():
    # Train model and get run ID
    run_id = train_and_log_model()
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Model trained and logged with URI: {model_uri}")
    
    # Save model URI to file
    with open("model_uri.txt", "w") as f:
        f.write(model_uri)
    
    # Deploy to Vertex AI (uncomment and add your project ID to deploy)
    # deploy_info = deploy_mlflow_model(
    #     project_id="your-gcp-project-id",
    #     region="us-central1",
    #     model_uri=model_uri,
    #     endpoint_name="iris-example"
    # )
    # 
    # print(f"Model deployed to Vertex AI endpoint: {deploy_info['endpoint_name']}")

if __name__ == "__main__":
    main() 