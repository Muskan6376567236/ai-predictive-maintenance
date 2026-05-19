from fastapi import FastAPI, HTTPException, Depends
from typing import List
from datetime import datetime
import logging

from src.utils.models import PredictionRequest, PredictionResponse, HealthCheck, FailurePrediction
from src.utils.config import Config
from src.prediction_engine.engine import PredictionEngine

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Predictive Maintenance API",
    description="API for predicting equipment failures and monitoring IoT sensor data",
    version="0.1.0"
)

# Load configuration
config = Config()

# Initialize prediction engine
prediction_engine = PredictionEngine()

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint."""
    return HealthCheck()

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_failure(request: PredictionRequest):
    """
    Predict failure for a specific piece of equipment based on sensor data.
    """
    logger.info(f"Received prediction request for equipment: {request.equipment_id}")
    
    if not request.sensor_data:
        raise HTTPException(status_code=400, detail="Sensor data is required for prediction")

    # Use prediction engine for actual inference
    prediction = prediction_engine.predict_failure(request.equipment_id, request.sensor_data)
    
    return PredictionResponse(
        equipment_id=request.equipment_id,
        prediction=prediction
    )

@app.get("/api/equipment/{equipment_id}")
async def get_equipment_details(equipment_id: str):
    """
    Retrieve current status and historical data for a specific equipment.
    """
    # Placeholder for database retrieval
    return {
        "equipment_id": equipment_id,
        "status": "operational",
        "last_maintenance": "2024-05-10",
        "next_scheduled_maintenance": "2024-08-10"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)