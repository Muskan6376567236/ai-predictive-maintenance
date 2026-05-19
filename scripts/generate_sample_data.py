import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(output_path="data/raw/sensor_data.csv", n_rows=1000):
    """Generate synthetic sensor data for testing."""
    np.random.seed(42)
    
    base_time = datetime.now()
    data = []
    
    for i in range(n_rows):
        timestamp = base_time + timedelta(minutes=i)
        # Normal operation with some noise
        temp = 70 + np.random.normal(0, 2)
        vibration = 1.0 + np.random.normal(0, 0.1)
        pressure = 50 + np.random.normal(0, 5)
        
        # Inject an anomaly towards the end
        if i > n_rows * 0.9:
            temp += (i - n_rows * 0.9) * 0.5
            vibration += (i - n_rows * 0.9) * 0.1
            
        data.append({
            "timestamp": timestamp,
            "equipment_id": "EQ_001",
            "temperature": temp,
            "vibration_x": vibration,
            "pressure": pressure
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_rows} rows of sample data at {output_path}")

if __name__ == "__main__":
    generate_sample_data()
