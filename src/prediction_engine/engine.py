"""Prediction engine for real-time equipment failure forecasting."""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.models import SensorReading, FailurePrediction
from src.model_training.architectures import LSTMPredictor
from src.feature_engineering.extractor import FeatureEngineer
from src.data_processing.processor import DataProcessor

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Real-time prediction engine that combines multiple models
    and feature engineering for accurate failure predictions.
    """

    def __init__(self, model_path: str = "models/trained_models/"):
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
        self.model: Optional[LSTMPredictor] = None
        self.is_loaded = False

    def load_model(self, model_name: str = "lstm_predictor.pt"):
        """Load a trained model from disk."""
        try:
            self.model = LSTMPredictor(
                input_dim=10,  # Will be updated based on actual number of features
                hidden_dim=64,
                num_layers=2
            )
            self.model.load_model(self.model_path + model_name)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False

    def process_sensor_data(self, readings: List[SensorReading]) -> np.ndarray:
        """
        Convert sensor readings into feature vectors for model input.
        """
        # Extract time-series data from readings
        timestamps = [r.timestamp for r in readings]
        temp_data = np.array([r.temperature if r.temperature else 0.0 for r in readings])
        vibration_x = np.array([r.vibration_x if r.vibration_x else 0.0 for r in readings])
        vibration_y = np.array([r.vibration_y if r.vibration_y else 0.0 for r in readings])
        vibration_z = np.array([r.vibration_z if r.vibration_z else 0.0 for r in readings])
        acoustic = np.array([r.acoustic if r.acoustic else 0.0 for r in readings])
        pressure = np.array([r.pressure if r.pressure else 0.0 for r in readings])
        rpm = np.array([r.rpm if r.rpm else 0.0 for r in readings])

        # Stack into feature matrix
        features = np.column_stack([
            temp_data,
            vibration_x, vibration_y, vibration_z,
            acoustic, pressure, rpm
        ])

        # Extract additional features
        extended_features = []
        for col_idx in range(features.shape[1]):
            col_data = features[:, col_idx]
            feat = self.feature_engineer.extract_all_features(col_data)
            # Flatten and add to row
            if feat:
                row_features = list(feat.values())
            else:
                row_features = [0.0] * 10
            extended_features.append(row_features)

        extended_features = np.array(extended_features).T  # transposed!

        # Combine original features with engineered features
        input_data = np.concatenate([features, extended_features], axis=1)
        return input_data[-30:]  # Return last 30 time steps as a window

    def predict_failure(self, equipment_id: str, sensor_history: List[SensorReading]) -> FailurePrediction:
        """
        Predict failure probability and remaining useful life for given equipment.
        """
        if not self.is_loaded or self.model is None:
            # Fallback to heuristic if model not loaded
            return self._heuristic_prediction(equipment_id, sensor_history)

        # Process sensor data
        input_window = self.process_sensor_data(sensor_history)

        # Prepare model input
        x = torch.from_numpy(input_window).float().unsqueeze(0)  # (1, seq_len, features)

        # Run inference
        with torch.no_grad():
            prediction, _ = self.model(x)
            prob = prediction.item()

        # Estimate time to failure (simplified)
        time_to_failure = max(1, int((1 - prob) * 1000))  # heuristic

        # Determine top contributing factors
        factors = self._identify_top_factors(sensor_history)

        confidence = 0.85 if prob > 0.5 else 0.92

        return FailurePrediction(
            equipment_id=equipment_id,
            probability=prob,
            time_to_failure_hours=time_to_failure,
            confidence=confidence,
            top_contributing_factors=factors
        )

    def _heuristic_prediction(self, equipment_id: str, sensor_history: List[SensorReading]) -> FailurePrediction:
        """Fallback heuristic prediction when model is unavailable."""
        if not sensor_history:
            return FailurePrediction(
                equipment_id=equipment_id,
                probability=0.0,
                time_to_failure_hours=9999,
                confidence=1.0,
                top_contributing_factors=["Insufficient data"]
            )

        # Simple baseline: check if temperature or vibration is high
        temps = [r.temperature for r in sensor_history if r.temperature]
        vibes = [r.vibration_x for r in sensor_history if r.vibration_x]

        avg_temp = np.mean(temps) if temps else 0
        max_vibe = np.max(vibes) if vibes else 0

        prob = 0.0
        if avg_temp > 85:
            prob += 0.3
        if max_vibe > 5.0:
            prob += 0.4
        if avg_temp > 90 and max_vibe > 7.0:
            prob += 0.3

        prob = min(prob, 0.99)

        return FailurePrediction(
            equipment_id=equipment_id,
            probability=prob,
            time_to_failure_hours=int((1 - prob) * 1000),
            confidence=0.7,
            top_contributing_factors=["High temperature" if avg_temp > 80 else "Normal operation"]
        )

    def _identify_top_factors(self, sensor_history: List[SensorReading]) -> List[str]:
        """
        Identify top contributing factors for anomaly.
        """
        if not sensor_history:
            return ["No data"]

        # Heuristic: find sensors with highest deviation from normal range
        factors = []

        # Temperature check
        temps = [r.temperature for r in sensor_history if r.temperature is not None]
        if temps and np.mean(temps) > 80:
            factors.append("Elevated temperature")

        # Vibration check
        vibes_x = [r.vibration_x for r in sensor_history if r.vibration_x is not None]
        if vibes_x and np.max(vibes_x) > 5.0:
            factors.append("High vibration")

        if not factors:
            factors.append("Normal operating conditions")

        return factors
