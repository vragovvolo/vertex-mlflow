import os
import logging
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sklearn.tree import DecisionTreeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Create a simple Iris classifier
class IrisModel:
    def __init__(self):
        logger.info("Initializing Iris model")
        self.clf = DecisionTreeClassifier(max_depth=3)
        
        # Sample Iris dataset for training
        X = [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],  # Setosa
            [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],  # Versicolor
            [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]   # Virginica
        ]
        # Classes: 0=setosa, 1=versicolor, 2=virginica
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        
        self.clf.fit(X, y)
        logger.info("Iris model training complete")
    
    def predict(self, instances):
        logger.info(f"Predicting for {len(instances)} instances")
        predictions = self.clf.predict(instances).tolist()
        logger.info(f"Predictions: {predictions}")
        return predictions

# Initialize model
model = IrisModel()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        instances = body.get("instances", [])
        logger.info(f"Received prediction request with {len(instances)} instances")
        
        if not instances:
            return JSONResponse(status_code=400, content={"error": "No instances provided"})
        
        predictions = model.predict(instances)
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 