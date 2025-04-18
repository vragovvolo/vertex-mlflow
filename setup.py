from setuptools import setup, find_packages

setup(
    name="vertex-mlflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "mlflow",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="MLflow integration for Google Cloud Vertex AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vertex-mlflow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "vertex-mlflow=vertex_mlflow.cli:main",
        ],
    },
) 