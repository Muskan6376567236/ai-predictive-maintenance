"""Data models for the predictive maintenance system."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


@dataclass
class SensorReading:
    """Represents a single sensor reading."""

    equipment_id: str
    timestamp: datetime
    temperature: Optional[float] = None
    vibration_x: Optional[float] = None
    vibration_y: Optional[float] = None
    vibration_z: Optional[float] = None
    acoustic: Optional[float] = None
    pressure: Optional[float] = None
    rpm: Optional[float] = None


@dataclass
class EquipmentStatus:
    """Current status of an equipment."""

    equipment_id: str
    status: str  # operational, warning, critical
    last_check: datetime
    days_since_maintenance: int
    hours_until_next_maintenance: Optional[int] = None


@dataclass
class FailurePrediction:
    """Prediction result for equipment failure."""

    equipment_id: str
    probability: float
    time_to_failure_hours: Optional[float]
    confidence: float
    top_contributing_factors: List[str]


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    equipment_id: str
    sensor_data: List[SensorReading]


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    equipment_id: str
    prediction: FailurePrediction
    timestamp: datetime = datetime.utcnow()


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"