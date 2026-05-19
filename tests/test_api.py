from fastapi.testclient import TestClient
from src.api.main import app
from datetime import datetime

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}

def test_predict_endpoint():
    """Test the prediction endpoint with mock data."""
    payload = {
        "equipment_id": "TEST-01",
        "sensor_data": [
            {
                "equipment_id": "TEST-01",
                "timestamp": str(datetime.now()),
                "temperature": 85.5,
                "vibration_x": 1.5
            }
        ]
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["equipment_id"] == "TEST-01"
    assert "prediction" in data
