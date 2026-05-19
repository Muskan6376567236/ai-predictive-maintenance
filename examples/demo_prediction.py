#!/usr/bin/env python3
"""
Demo script showing how to use the AI Predictive Maintenance System
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.models import SensorReading
from src.prediction_engine.engine import PredictionEngine
from src.api.main import app

def demo_prediction():
    """Demonstrate a simple prediction using the prediction engine."""
    print("🤖 AI Predictive Maintenance System Demo")
    print("=" * 50)
    
    # Initialize prediction engine
    engine = PredictionEngine()
    
    # Create sample sensor data (simulating normal operation)
    base_time = datetime.now()
    sensor_history = []
    
    # Generate 30 time steps of sensor data
    for i in range(30):
        reading = SensorReading(
            equipment_id="PUMP-001",
            timestamp=base_time - timedelta(minutes=30-i),
            temperature=75.0 + (i * 0.2),  # Gradually increasing temperature
            vibration_x=1.2 + (i * 0.05),   # Increasing vibration
            vibration_y=0.8,
            vibration_z=1.0,
            acoustic=0.3,
            pressure=45.0,
            rpm=1800
        )
        sensor_history.append(reading)
    
    # Make prediction
    print(f"\n📊 Analyzing {len(sensor_history)} sensor readings for equipment PUMP-001")
    print("   Temperature range: {:.1f}°C to {:.1f}°C".format(
        min(r.temperature for r in sensor_history if r.temperature),
        max(r.temperature for r in sensor_history if r.temperature)
    ))
    print("   Vibration X range: {:.2f} to {:.2f} mm/s".format(
        min(r.vibration_x for r in sensor_history if r.vibration_x),
        max(r.vibration_x for r in sensor_history if r.vibration_x)
    ))
    
    prediction = engine.predict_failure("PUMP-001", sensor_history)
    
    # Display results
    print(f"\n🎯 PREDICTION RESULTS")
    print(f"   Equipment ID: {prediction.equipment_id}")
    print(f"   Failure Probability: {prediction.probability:.1%}")
    print(f"   Time to Failure: {prediction.time_to_failure_hours:.1f} hours")
    print(f"   Confidence: {prediction.confidence:.1%}")
    print(f"   Top Contributing Factors: {', '.join(prediction.top_contributing_factors)}")
    
    # Risk assessment
    if prediction.probability > 0.7:
        risk_level = "🔴 HIGH"
    elif prediction.probability > 0.3:
        risk_level = "🟡 MEDIUM"
    else:
        risk_level = "🟢 LOW"
        
    print(f"\n⚠️  Risk Assessment: {risk_level}")
    
    return prediction

if __name__ == "__main__":
    demo_prediction()
