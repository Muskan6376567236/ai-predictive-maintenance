# AI-Powered Predictive Maintenance System

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0 License">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Active">
  <img src="https://img.shields.io/badge/architecture-modular-orange.svg" alt="Modular Architecture">
  <img src="https://img.shields.io/github/stars/Muskan6376567236/ai-predictive-maintenance?style=social" alt="GitHub stars">
</div>


# AI-Powered Predictive Maintenance System

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0 License">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Active">
  <img src="https://img.shields.io/badge/architecture-modular-orange.svg" alt="Modular Architecture">
</div>

## Overview

An advanced predictive maintenance system that uses machine learning to predict equipment failures before they occur. This system combines IoT sensor data with state-of-the-art deep learning models to provide accurate failure predictions, reducing downtime and maintenance costs.

Built with production-grade reliability, featuring multi-modal sensor fusion, real-time anomaly detection, and explainable AI insights.

## Features

- Real-time equipment monitoring via sensor data
- Advanced failure prediction algorithms (LSTM, Transformers, Hybrid CNN-LSTM)
- Anomaly detection for early warning systems
- Customizable alert thresholds and notification channels
- Comprehensive dashboard for visualization and analysis
- RESTful API for integration with existing systems
- Multi-modal data processing (vibration, temperature, acoustic, etc.)
- Explainable AI for decision transparency

## Project Structure

```
ai-predictive-maintenance/
├── data/
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Cleaned and processed data
│   └── external/               # Third-party datasets
├── models/
│   ├── trained_models/         # Saved model weights
│   └── training_scripts/       # Model training utilities
├── src/
│   ├── __init__.py
│   ├── data_processing/        # Data ingestion and cleaning
│   ├── feature_engineering/    # Feature extraction and selection
│   ├── model_training/         # Model training pipelines
│   ├── prediction_engine/      # Real-time prediction service
│   ├── api/                    # FastAPI endpoints
│   └── utils/                  # Common utilities
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks for exploration
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- PostgreSQL (for data storage)
- Redis (for real-time processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/Muskan6376567236/ai-predictive-maintenance.git
cd ai-predictive-maintenance

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Update the configuration with your local settings
3. Set up environment variables as specified

## Data Preparation

### Sensor Data Requirements

- Vibration measurements (accelerometer data)
- Temperature readings
- Acoustic emissions
- Operational parameters (RPM, pressure, etc.)
- Historical failure records

### Data Processing Pipeline

1. Raw data ingestion from sensors/IoT devices
2. Data cleaning and normalization
3. Feature extraction (statistical, frequency domain, time domain)
4. Labeling (for supervised learning)
5. Train-test split

## Model Training

### Available Models

1. LSTM-based Time Series Model - for temporal patterns
2. Transformer-based Model - for long-range dependencies
3. Hybrid CNN-LSTM Model - for multi-scale feature extraction
4. Isolation Forest - for unsupervised anomaly detection

### Training Process

```bash
# Train the default LSTM model
python src/model_training/train_lstm.py --config config/local.yaml

# Train the transformer model
python src/model_training/train_transformer.py

# Train all available models
bash scripts/train_all_models.sh
```

## Prediction Engine

### Running Predictions

```bash
# Run prediction engine
python src/prediction_engine/main.py --config config/local.yaml
```

### API Endpoints

```
POST /api/predict          - Get predictions for new sensor data
GET  /api/status           - Check system status
GET  /api/equipment/{id}   - Get equipment-specific predictions
GET  /api/alerts           - Get active alerts
POST /api/train            - Trigger model retraining
```

### Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"equipment_id": "EQ001", "sensor_data": [...]}
)
print(response.json())
```

## Dashboard

### Running the Dashboard

```bash
# Start the dashboard
uvicorn src.api.main:app --reload --port 8000
```

### Dashboard Features

- Real-time equipment status monitoring
- Historical failure trends visualization
- Prediction confidence scores
- Alert management interface
- Custom report generation

## Deployment

### Docker Deployment

```bash
# Build the Docker image
docker build -t predictive-maintenance .

# Run the container
docker run -p 8000:8000 --env-file .env predictive-maintenance
```

### Kubernetes Deployment

For production deployment, see `kubernetes/` directory for configuration files.

## Monitoring and Alerts

### Alert Configuration

Alert thresholds can be configured in `config/alerts.yaml`. The system supports:

- Email alerts
- SMS alerts
- Webhook notifications
- Slack/Teams integration

### Monitoring Setup

```bash
# Start the monitoring system
python src/monitoring/main.py --config config/local.yaml
```

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for more details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Implement multi-modal data fusion
- [ ] Add explainable AI features (SHAP/LIME)
- [ ] Develop mobile application
- [ ] Implement edge computing capabilities
- [ ] Add support for additional industrial protocols

## Support

For support, please open an issue on GitHub or contact our support team at support@predictivemaintenance.ai.

---

<div align="center">
  Made with ❤️ by the Predictive Maintenance Team
</div>